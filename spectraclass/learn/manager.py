import xarray as xa
import pandas as pd
import time, traceback, abc
from functools import partial
from jupyter_bokeh.widgets import BokehModel
import numpy as np
import scipy, sklearn
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict, Union
from bokeh.models import ColumnDataSource, DataTable, TableColumn, Selection
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
import ipywidgets as ipw
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable
from .base import LearningModel, KerasModelWrapper

def cm():
    return ClassificationManager.instance()

class ModelTable:

    def __init__(self, data: Union[pd.DataFrame,List[str]], **kwargs ):
        self._dataFrame: pd.DataFrame = None
        if isinstance( data, pd.DataFrame ):
            self._dataFrame = data
        elif isinstance( data, List ):
            lgm().log( f"Creating DataFrame from list: {data}")
            self._dataFrame = pd.DataFrame( data, columns=["models"] )
        else:
            raise TypeError( f"Unsupported data class supplied to ModelTable: {data.__class__}" )
        self._source: ColumnDataSource = ColumnDataSource( self._dataFrame )
        self._columns = [ TableColumn(field=cid, title=cid) for cid in self._dataFrame.columns ]
        self._table = DataTable( source=self._source, columns=self._columns ) # , width=500, height=500, selectable="checkbox" )

    def to_df( self ) -> pd.DataFrame:
        return self._source.to_df()

    def delete(self, model_index ):
        lgm().log( f" ModelTable delete index: {model_index}" )

    def selected_row( self ):
        column: pd.Series = self._dataFrame["models"]
        return column.values[ self.selection ]

    @property
    def selection( self ) -> List[int]:
        return self._source.selected.indices

    @exception_handled
    def gui(self) -> ipw.DOMWidget:
        return BokehModel(self._table) # ipw.HBox( [ BokehModel(self._table) ] )

class Cluster:

    def __init__(self, cid, **kwargs):
        self.cid = cid
        self._members = []
        self.metrics = {}

    def addMember(self, example: np.ndarray ):
        self._members.append( example )
        self.metrics = {}

    @property
    def members(self) -> np.ndarray:
        return np.vstack(self._members)

    @property
    def mean(self):
        if "mean" not in self.metrics.keys():
            self.metrics["mean"] = self.members.mean(0)
        return self.metrics["mean"]

    @property
    def std(self):
        if "std" not in self.metrics.keys():
            self.metrics["std"] = self.members.std(0)
        return self.metrics["std"]

    @property
    def cov(self):
        if "cov" not in self.metrics.keys():
            self.metrics["cov"] = np.cov( self.members.transpose() )
        return self.metrics["cov"]

    @property
    def cor(self):
        if "cor" not in self.metrics.keys():
            self.metrics["cor"] = np.corrcoef( self.members.transpose() )
        return self.metrics["cor"]

    @property
    def icov(self):
        if "icov" not in self.metrics.keys():
            self.metrics["icov"] = scipy.linalg.pinv(self.cov)
        return self.metrics["icov"]

    @property
    def icor(self):
        if "icor" not in self.metrics.keys():
            self.metrics["icor"] = scipy.linalg.pinv(self.cor)
        return self.metrics["icor"]

class ClassificationManager(SCSingletonConfigurable):
    mid = tl.Unicode("mlp").tag(config=True, sync=True)

    def __init__(self,  **kwargs ):
        super(ClassificationManager, self).__init__(**kwargs)
        self._models: Dict[str,LearningModel] = {}
        self.import_models()
        self.selection = self.selection_label = None
        self.model_table: ModelTable = None

    @property
    def mids(self) -> List[str]:
        return list(self._models.keys())

    @exception_handled
    def create_selection_panel(self, **kwargs ):
        default = kwargs.get( 'default', self.mids[0] )
        self.selection_label = ipw.Label(value='Learning Model:')
        self.selection = ipw.RadioButtons(  options=self.mids, disabled=False, layout={'width': 'max-content'}, value=default )
        self.selection.observe( self.set_model_callback, "value" )
        self.mid = default

    def set_model_callback(self, event: Dict ):
        self.mid = event['new']

    def import_models(self):
        from .svc import SVCLearningModel
        self._models['mlp'] = self.create_default_mlp()
        self._models['svc'] = SVCLearningModel()
        self._models['cnn'] = self.create_default_cnn()

    def create_default_cnn(self) -> "LearningModel":
        pass

    def create_default_mlp(self) -> "LearningModel":
        from .mlp import MLP
        from spectraclass.model.labels import lm
        from spectraclass.data.base import DataManager, dm
        from .base import KerasModelWrapper
        return KerasModelWrapper( "mlp", MLP.build( dm().modal.model_dims, lm().nLabels ) )

    def addLearningModel(self, mid: str, model: "LearningModel" ):
        self._models[ mid ] = model

    def addNNModel(self, mid: str, model: Model, **kwargs):
        self._models[ mid ] = KerasModelWrapper(mid, model, **kwargs)

    def get_control_button(self, task: str ) -> ipw.Button:
        button = ipw.Button(description=task, border='1px solid gray')
        button.layout = ipw.Layout(width='auto', flex="1 0 auto")
        button.on_click( partial(self.on_control_click, task) )
        return button

    @exception_handled
    def create_persistence_gui(self) -> ipw.DOMWidget:
        title = ipw.Label(value="Persisted Models")
        self.model_table = ModelTable( self.model.list_models() )
        controls = [ self.get_control_button(task) for task in [ "save", "load", "delete" ] ]
        mlist = self.model_table.gui() # ] ) # , ipw.HBox( controls ) ] ) # , layout = ipw.Layout( width="500px", height="500px", border= '2px solid firebrick' )  )
        gui = ipw.VBox([ title, mlist, ipw.HBox( controls ) ] )  # , layout = ipw.Layout( width="500px", height="500px", border= '2px solid firebrick' )  )
        return gui

    @exception_handled
    def on_control_click( self, task, button: ipw.Button = None ):
        if   ( task == "save" ):
            self.save_model( )
        elif ( task == "load" ):
            selected_model = self.model_table.selected_row()
            lgm().log( f"MODEL TABLE <LOAD>: Selected model: {selected_model}")
            cm().load_model( selected_model )
        elif ( task == "delete" ):
            model_index = self.model_table.selection
            lgm().log(f"MODEL TABLE <DEL>: Selected index: {model_index}")
            self.model_table.delete( model_index )

    def gui(self):
        if self.selection is None: self.create_selection_panel()
        return ipw.HBox( [self.selection_label, self.selection] )
        # distanceMetric = base.createComboSelector("Distance.Metric: ", ["mahal","euclid"], "dev/distance/metric", "mahal")
        # distanceMethod = base.createComboSelector("Distance.Method: ", ["centroid","nearest"], "dev/distance/method", "centroid")
        # return base.createGroupBox("dev", [model, distanceMetric, distanceMethod ] )

    @property
    def model(self) -> "LearningModel":
        model: LearningModel = self._models[ self.mid ]
        return model

    def save_model( self, **kwargs ):
        self.model.save( **kwargs )

    def load_model( self, model_name, **kwargs ):
        self.model.load( model_name, **kwargs )

    @exception_handled
    def learn_classification( self, filtered_point_data: np.ndarray, filtered_labels: np.ndarray, **kwargs  ):
        lgm().log( f"\n learn_classification-> point_data: {filtered_point_data.shape}, labels: {filtered_labels.shape} \n")
        self.model.learn_classification( filtered_point_data, filtered_labels, **kwargs  )

    @exception_handled
    def apply_classification( self, embedding: xa.DataArray, **kwargs ) -> xa.DataArray:
        try:
            ufm().show("Applying Classification... ")
            sample_labels: xa.DataArray = self.model.apply_classification( embedding, **kwargs  )
            return sample_labels
        except sklearn.exceptions.NotFittedError:
            ufm().show( "Must learn a mapping before applying a classification", "red")

