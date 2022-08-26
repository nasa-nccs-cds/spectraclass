import xarray as xa
import pandas as pd
import time, traceback, shutil
import ipysheet as ips
from functools import partial
import numpy as np
import scipy, sklearn
import tensorflow as tf
keras = tf.keras
from keras.models import Model
from typing import List, Tuple, Optional, Dict, Union
import traitlets as tl
import traitlets.config as tlc
import ipywidgets as ipw
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable
from .base import LearningModel

def cm():
    return ClassificationManager.instance()

class ModelTable:

    def __init__(self, models: Dict[str,str], **kwargs ):
        self._models = models
        self._cdata = dict( models=list(models.keys()) )
        self._cnames = list( self._cdata.keys() )
        self._mulitselect = kwargs.get( 'mulitselect', False )
        self._callbacks_active = True
        self._selections_cell = None
        self._dataFrame = pd.DataFrame( self._cdata, columns=self._cnames )
        lgm().log(f"Creating ModelTable from DataFrame: {self._dataFrame}")
        self._table: ips.Sheet = ips.Sheet( rows=100, columns=len(self._cnames)+1,
                                         cells=self.get_table_cells(), row_headers=False, column_headers=[""]+self._cnames )

    def to_df( self ) -> pd.DataFrame:
        return self._dataFrame

    def cell(self, data: List, col: int, type: str, **kwargs ):
        cell = ips.Cell( value=data, row_start=0, row_end=len(data)-1, column_start=col, column_end=col, type=type,
                         read_only=kwargs.get('read_only',False), squeeze_row=False, squeeze_column=True )
        observer = kwargs.get( 'observer', None )
        if observer is not None: cell.observe( observer, 'value' )
        return cell

    def get_table_cells(self):
        lgm().log( f"Refreshing table, data = {self._dataFrame}")
        if self._dataFrame.shape[0] == 0: return []
        selections_init = [ False ] * self._dataFrame.shape[0]
        cells = [ self.cell( self._dataFrame[c].values.tolist(), idx+1, 'text', read_only=(idx==0) ) for idx, c in enumerate(self._cnames) ]
        self._selections_cell = self.cell( selections_init, 0, 'checkbox', observer=self.on_selection_change )
        cells.append( self._selections_cell )
        return cells

    @exception_handled
    def on_selection_change( self, change: Dict ):
        if self._callbacks_active:
            if not self._mulitselect:
                oldv, newv = np.array( change['old'] ), np.array( change['new'] )
                change_indices = np.where( ~( oldv == newv ) )[0]
                change_value = newv[ change_indices[0] ]
                if change_value:
                    self._callbacks_active = False
                    selections = [False] * self._dataFrame.shape[0]
                    selections[ change_indices[0] ] = True
                    self._selections_cell.value = selections
                    self._callbacks_active = True
                lgm().log( f"ModelTable: selection: {self.selection}" )

    def refresh(self):
        self._table.cells = self.get_table_cells()

    @exception_handled
    def add(self, model_name: str ):
        self._dataFrame = self._dataFrame.append( pd.DataFrame( [model_name], columns=["models"] ), ignore_index=True )
        self.refresh()

    @property
    def index(self) -> List[int]:
        return self._dataFrame.index.tolist()

    @exception_handled
    def delete_model_file(self, model_name: str ):
        mdir: str = self._models[ model_name ]
        lgm().log( f" Deleting model dir '{mdir}' ")
        shutil.rmtree( mdir )

    def delete(self, row: int ):
        lgm().log(f" Deleting row '{row}', dataFrame index = {self.index} ")
        idx: int = self.index[row]
        column: pd.Series = self._dataFrame["models"]
        self.delete_model_file( column.values[ row ] )
        self._dataFrame = self._dataFrame.drop( index=idx )
        self.refresh()

    def selected_row( self ):
        column: pd.Series = self._dataFrame["models"]
        return column.values[ self.selection ]

    @property
    def selection( self ) -> List[int]:
        return np.where( self._selections_cell.value )[0].tolist()

    @exception_handled
    def gui(self) -> ipw.DOMWidget:
        return ipw.HBox( [ self._table ], layout = ipw.Layout( width="300px", height="300px", border= '2px solid firebrick' ) )

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
    nfeatures =  tl.Int(32).tag(config=True, sync=True)

    def __init__(self,  **kwargs ):
        super(ClassificationManager, self).__init__(**kwargs)
        self._models: Dict[str,LearningModel] = {}
        self.import_models()
        self.selection = self.selection_label = None
        self.model_table: ModelTable = None

    def rebuild(self):
        for lmodel in self._models.values(): lmodel.rebuild()

    @property
    def mids(self) -> List[str]:
        return list(self._models.keys())

    def addModel(self, name: str, model: LearningModel ):
        self._models[ name ] = model

    @exception_handled
    def create_selection_panel(self, **kwargs ):
        self.selection_label = ipw.Label(value='Learning Model:')
        self.selection = ipw.RadioButtons(  options=self.mids, disabled=False, layout={'width': 'max-content'}, value=self.mid )
        self.selection.observe( self.set_model_callback, "value" )

    def set_model_callback(self, event: Dict ):
        self.mid = event['new']

    def import_models(self):
        from .cnn import CNN
        from .svc import SVCLearningModel
        from .mlp import MLP
        from spectraclass.data.base import DataManager, dm
        lgm().log("#IA: INIT MODELS")
        self.addNetwork( MLP( 'mlp', nfeatures=dm().modal.model_dims) )
        self.addNetwork( CNN( 'cnn', nfeatures=self.nfeatures ) )
        self._models['svc'] = SVCLearningModel()

    def addLearningModel(self, mid: str, model: LearningModel ):
        self._models[ mid ] = model

    def addNNModel(self, mid: str, model: Model, **kwargs):
        from spectraclass.learn.base import KerasLearningModel
        self._models[ mid ] = KerasLearningModel(mid, model, **kwargs)

    def addNetwork(self, network ):
        lgm().log( f"#IA: ADD NETWORK MODEL: {network.name}" )
        self._models[ network.name ] = network.build()

    def get_control_button(self, task: str ) -> ipw.Button:
        button = ipw.Button(description=task, border='1px solid gray')
        button.layout = ipw.Layout(width='auto', flex="1 0 auto")
        button.on_click( partial(self.on_control_click, task) )
        return button

    @exception_handled
    def create_persistence_gui(self) -> ipw.DOMWidget:
        title = ipw.Label( value="Persisted Models" )
        self.model_table = ModelTable( self.model.list_models() )
        controls = [ self.get_control_button(task) for task in [ "save", "load", "delete" ] ]
        mtable = self.model_table.gui() # ] ) # , ipw.HBox( controls ) ] ) # , layout = ipw.Layout( width="500px", height="500px", border= '2px solid firebrick' )  )
        gui = ipw.VBox( [ title, mtable, ipw.HBox( controls ) ]  )
        return gui

    @exception_handled
    def on_control_click( self, task, button: ipw.Button = None ):
        if   ( task == "save" ):
            model_name = self.save_model( )
            self.model_table.add( model_name )
        elif ( task == "load" ):
            selected_model = self.model_table.selected_row()
            if len( selected_model ):
                lgm().log( f"MODEL TABLE <LOAD>: Selected model: {selected_model[0]}")
                ufm().show( f"Loading model {selected_model[0]}")
                cm().load_model( selected_model[0] )
        elif ( task == "delete" ):
            model_index = self.model_table.selection
            if len( model_index ):
                lgm().log(f"MODEL TABLE <DEL>: Selected index: {model_index[0]}")
                self.model_table.delete( model_index[0] )

    def gui(self):
        from spectraclass.model.labels import LabelsManager, lm
        if self.selection is None: self.create_selection_panel()

        clear_button = ipw.Button(description='Reset', border='1px solid gray')
        clear_button.layout = ipw.Layout(width='auto', flex="1 0 auto")
        clear_button.on_click( self.model.clear )

        save_button = ipw.Button(description='Save Labels', border='1px solid gray')
        save_button.layout = ipw.Layout(width='auto', flex="1 0 auto")
        save_button.on_click( lm().saveLabelData )

        buttonbox = ipw.VBox( [clear_button,save_button] )
        return ipw.HBox( [self.selection_label, self.selection, buttonbox ] )
        # distanceMetric = base.createComboSelector("Distance.Metric: ", ["mahal","euclid"], "dev/distance/metric", "mahal")
        # distanceMethod = base.createComboSelector("Distance.Method: ", ["centroid","nearest"], "dev/distance/method", "centroid")
        # return base.createGroupBox("dev", [model, distanceMetric, distanceMethod ] )

    @property
    def model(self) -> "LearningModel":
        model: LearningModel = self._models[ self.mid ]
        return model

    def save_model( self, **kwargs ) -> str:
        return self.model.save( **kwargs )

    def load_model( self, model_name, **kwargs ):
        self.model.load( model_name, **kwargs )

    @exception_handled
    def learn_classification( self, **kwargs  ):
        lgm().log( f"learn_classification: MODEL = {self.mid} ")
        self.model.learn_classification( **kwargs  )

    @exception_handled
    def apply_classification( self, **kwargs  ):
        lgm().log( f"apply_classification: MODEL({hex(id(self.model))}) = {self.mid} ")
        self.model.apply_classification( **kwargs  )


