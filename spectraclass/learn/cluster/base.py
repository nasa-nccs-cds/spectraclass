from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import numpy as np
import xarray as xa

class  ClusterBase(TransformerMixin,ClusterMixin,BaseEstimator):

    def __init__( self, n_clusters: int ):
        self._n_clusters = n_clusters

    def cluster( self, data: xa.DataArray, y=None ) -> np.ndarray:
        raise NotImplementedError( "Call to abstract method of base class: ClusterBase.fit_predict")

    def _update_nclusters( self ):
        raise NotImplementedError( "Call to abstract method of base class: ClusterBase._update_nclusters")

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @n_clusters.setter
    def n_clusters( self, value: int ):
        self._n_clusters = value
        self._update_nclusters()





