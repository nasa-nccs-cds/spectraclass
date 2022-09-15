from spectraclass.util.logs import LogManager, lgm, log_timing
from sklearn import cluster
import xarray as xa
import numpy as np
from .base import GenericClusterBase
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type, Iterable

class KMeansCluster(GenericClusterBase):
    def update_model(self):
        params = dict( n_clusters=self._n_clusters, random_state=self.random_state, compute_labels=True, batch_size=self.batch_size )
        self._model = cluster.MiniBatchKMeans(**params)

class BisectingKMeans(GenericClusterBase):
    def update_model(self):
        params = dict( n_clusters=self._n_clusters, random_state=self.random_state, algorithm= "elkan", init="k-means++") # , bisecting_strategy="largest_cluster" )
        self._model = cluster.BisectingKMeans(**params)
