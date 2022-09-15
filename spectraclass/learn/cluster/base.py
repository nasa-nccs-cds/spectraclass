from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from spectraclass.util.logs import LogManager, lgm
from joblib import cpu_count
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type, Iterable
import numpy as np
import xarray as xa

class  ClusterBase(TransformerMixin,ClusterMixin,BaseEstimator):

    def __init__( self, n_clusters: int ):
        self._n_clusters = n_clusters
        self.cscale = np.full([1, self._n_clusters], 0.5)
        self._threshold = 0.0
        self._threshold_mask = None

    @property
    def cluster_data(self) -> xa.DataArray:
        raise NotImplementedError("Call to abstract property of base class: ClusterBase.cluster_data")

    def rescale(self, index: int, sval: float ):
        if index > 0:
            lgm().log(f"Cluster[{index}].rescale: value = {sval}")
            self.cscale[ 0, index ] = sval
        else:
            self._threshold = sval
        self._threshold_mask = None

    def reset(self):
        pass

    def cluster( self, data: xa.DataArray, y=None ):
        raise NotImplementedError( "Call to abstract method of base class: ClusterBase.fit_predict")

    def _update_nclusters( self ):
        pass

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @n_clusters.setter
    def n_clusters( self, value: int ):
        self._n_clusters = value
        self.cscale = np.full([1, self._n_clusters], 0.5)
        self._update_nclusters()


class GenericClusterBase(ClusterBase):

    def __init__( self, n_clusters: int, **kwargs ):
        ClusterBase.__init__( self, n_clusters )
        self.random_state = kwargs.get( 'random_state', 100 )
        self.batch_size = kwargs.get('batch_size', 256 * cpu_count() )
        self._cluster_data: np.ndarray = None
        self._cluster_distances: Dict[int, Tuple[np.ndarray,np.ndarray] ] = {}
        self._samples: np.ndarray = None
        self._attrs = None
        self._model = None
        self.update_model()

    def reset(self):
        self._cluster_data = None
        self._samples = None
        self._attrs = None

    def update_model(self):
        raise NotImplementedError("Attempt to call abstract method")

    @property
    def cluster_data(self) -> xa.DataArray:
        cdata = self._cluster_data.squeeze()
        if self._threshold > 0.0:  cdata = np.where( self.threshold_mask, cdata, 0 )
        lgm().log( f"cluster_data: data.shape={self._cluster_data.shape}, threshold-mask.shape = {self.threshold_mask.shape}"
                   f", threshold-mask.gtz = {np.count_nonzero(self.threshold_mask)}, cdata.shape = {cdata.shape},  #gtz = {np.count_nonzero(cdata)}")
        cdata_array = xa.DataArray( cdata, dims=['samples'], name="cluster_data", coords=dict( samples=self._samples ), attrs=self._attrs )
        return cdata_array.expand_dims( "cluster", 1 )

    def cluster( self, data: xa.DataArray, y=None ):
        self._attrs = data.attrs
        self._threshold_mask = None
        self._cluster_data = np.expand_dims( self._model.fit_predict( data.values ), axis=1 ) + 1
        self._samples = data.samples.values
        cluster_centers: np.ndarray = self._model.cluster_centers_

        for iC in range( 1, self.n_clusters+1 ):
            cmask = (self._cluster_data == iC).flatten()
            indices = data.samples.values[cmask]
            cluster_distance = np.linalg.norm( data.values[cmask,:] - cluster_centers[iC-1,:], axis=1 )
            dmask = np.isin( self._samples, indices, assume_unique=True )
            self._cluster_distances[iC] = ( dmask, cluster_distance )

    @property
    def threshold_mask( self ):
        if self._threshold_mask is None:
            cluster_distances = np.zeros( self._cluster_data.shape[0], np.float ) # ndarray[n-clusters,n-model-dims]
            for iC in range( 1, self.n_clusters+1 ):
                (dmask, cluster_distance) = self._cluster_distances[iC]
                cluster_distances[ dmask ] = cluster_distance / max( self.cscale[0,iC-1], 0.1 )
            uncertainty: np.ndarray = cluster_distances / cluster_distances.max()
            self._threshold_mask = ((1.0 - uncertainty) ) > self._threshold
        return self._threshold_mask

    @property
    def centers(self) -> np.ndarray:
        return self._model.cluster_centers_

    def _update_nclusters( self ):
        self.update_model()







