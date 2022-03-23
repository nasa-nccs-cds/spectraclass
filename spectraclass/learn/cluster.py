from sklearn import cluster
from sklearn.base import ClusterMixin
from joblib import cpu_count
import time, traceback, shutil
import ipywidgets as ipw
from functools import partial
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import traitlets as tl
import traitlets.config as tlc
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable

class ClusterManager(SCSingletonConfigurable):
    mid = tl.Unicode("kmeans").tag(config=True, sync=True)
    nclusters = tl.Int(12).tag(config=True, sync=True)
    random_state = tl.Int(0).tag(config=True, sync=True)

    def __init__(self,  **kwargs ):
        super(ClusterManager, self).__init__(**kwargs)
        self._ncluster_options = range( 6, 60, 3 )
        self._models: Dict[str,ClusterMixin] = {}
        self.import_models()
        self._model_selector = None
        self._ncluster_selector = None

    def import_models(self):
        for mid in [ 'kmeans' ]:
            self._models[mid] = self.create_default_model( mid )

    @property
    def mids(self) -> List[str]:
        return list( self._models.keys() )

    def create_default_model(self, mid: str ) -> ClusterMixin:
        if mid == "kmeans":
            params = dict( n_clusters= self._ncluster_selector.value,
                           random_state= self.random_state,
                           batch_size= 256 * cpu_count() )
            return cluster.MiniBatchKMeans( **params )

    @property
    def model(self) -> ClusterMixin:
        return self._models[ self._model_selector.value ]

    def cluster(self, data: np.ndarray ) -> np.ndarray:
        return self.model.fit_predict( data )

    @exception_handled
    def gui(self) -> ipw.DOMWidget:
        self._model_selector = ipw.Select(options=self.mids, description='Methods:', value=self.mid, disabled=False,
                                          layout=ipw.Layout(width="500px"))
        self._ncluster_selector = ipw.Select(options=self._ncluster_options, description='#Clusters:', disabled=False,
                                             value=self.nclusters, layout=ipw.Layout(width="500px"))
        return ipw.HBox([self._model_selector,self._ncluster_selector], layout=ipw.Layout(width="500px", height="600px", border='2px solid firebrick'))
