import xarray as xa
import time, traceback, abc
import numpy as np
import scipy, sklearn
from typing import List, Tuple, Optional, Dict
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable

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
    mid = tl.Unicode("svc").tag(config=True, sync=True)

    def __init__(self,  **kwargs ):
        super(ClassificationManager, self).__init__(**kwargs)
        self._models: Dict[str,LearningModel] = {}
        self.import_models()

    def import_models(self):
        from .svc import SVCLearningModel
        self._models['svc'] = SVCLearningModel()

    @property
    def mids(self):
        return [ m.mid for m in self._models.values() ]

    def addModel(self, mid: str, model: "LearningModel" ):
        self._models[ mid ] = model

    def gui(self):
        mids = self.mids
        # model = base.createComboSelector( "Model: ", mids, "dev/model", "cluster" )
        # distanceMetric = base.createComboSelector("Distance.Metric: ", ["mahal","euclid"], "dev/distance/metric", "mahal")
        # distanceMethod = base.createComboSelector("Distance.Method: ", ["centroid","nearest"], "dev/distance/method", "centroid")
        # return base.createGroupBox("dev", [model, distanceMetric, distanceMethod ] )

    @property
    def model(self) -> "LearningModel":
        model: LearningModel = self._models[ self.mid ]
        return model

    @exception_handled
    def learn_classification( self, filtered_point_data: np.ndarray, filtered_labels: np.ndarray, **kwargs  ):
        self.model.learn_classification( filtered_point_data, filtered_labels, **kwargs  )
        ufm().show( "Classification Mapping learned" )

    @exception_handled
    def apply_classification( self, embedding: xa.DataArray, **kwargs ) -> xa.DataArray:
        try:
            sample_labels: xa.DataArray = self.model.apply_classification( embedding, **kwargs  )
            return sample_labels
        except sklearn.exceptions.NotFittedError:
            ufm().show( "Must learn a mapping before applying a classification", "red")


class LearningModel:

    def __init__(self, name: str,  **kwargs ):
        self.mid =  name
        self._score: Optional[np.ndarray] = None
        self.config = kwargs
        self._keys = []

    def setKeys(self, keys: List[str] ):
        self._keys = keys

    @property
    def score(self) -> Optional[np.ndarray]:
        return self._score

    def learn_classification( self, point_data: np.ndarray, labels: np.ndarray, **kwargs  ):
        t1 = time.time()
        if np.count_nonzero( labels > 0 ) == 0:
            ufm().show( "Must label some points before learning the classification" )
            return None
        self.fit( point_data, labels, **kwargs )
        print(f"Learned mapping with {labels.shape[0]} labels in {time.time()-t1} sec.")

    def fit(self, data: np.ndarray, labels: np.ndarray, **kwargs):
        raise Exception( "abstract method LearningModel.fit called")

    def apply_classification( self, data: xa.DataArray, **kwargs ) -> xa.DataArray:
        t1 = time.time()
        prediction: np.ndarray = self.predict( data.values, **kwargs )
        print(f"Applied classication with input shape {data.shape[0]} in {time.time() - t1} sec.")
        return xa.DataArray( prediction, dims=['samples'], coords=dict( samples=data.coords['samples'] ) )

    def predict( self, data: np.ndarray, **kwargs ):
        raise Exception( "abstract method LearningModel.predict called")



