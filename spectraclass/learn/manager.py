import xarray as xa
import time, traceback, abc
import numpy as np
import scipy, sklearn
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
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
    mid = tl.Unicode("").tag(config=True, sync=True)

    def __init__(self,  **kwargs ):
        super(ClassificationManager, self).__init__(**kwargs)
        self._models: Dict[str,LearningModel] = {}
        self.import_models()
        self.selection = self.selection_label = None

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
        from .base import KerasModelWrapper
        mlp = MLP()
        return KerasModelWrapper( "mlp", mlp )

    def addLearningModel(self, mid: str, model: "LearningModel" ):
        self._models[ mid ] = model

    def addNNModel(self, mid: str, model: Model, **kwargs):
        self._models[ mid ] = KerasModelWrapper(mid, model, **kwargs)

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

