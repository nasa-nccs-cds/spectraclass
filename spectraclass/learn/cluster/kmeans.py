from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn import cluster
from joblib import cpu_count
import xarray as xa
import numpy as np
from .base import ClusterBase

class KMeansCluster(ClusterBase):

    def __init__( self, n_clusters: int, **kwargs ):
        ClusterBase.__init__( self, n_clusters )
        self.random_state = kwargs.get( 'random_state', 100 )
        self.batch_size = kwargs.get('batch_size', 256 * cpu_count() )
        self._cluster_data = None
        self._samples = None
        self._attrs = None
        self.update_model()

    def reset(self):
        self._cluster_data = None
        self._samples = None
        self._attrs = None

    def update_model(self):
        params = dict( n_clusters=self._n_clusters, random_state=self.random_state, batch_size=self.batch_size )
        self._model = cluster.MiniBatchKMeans(**params)

    @property
    def cluster_data(self) -> xa.DataArray:
        return xa.DataArray( self._cluster_data, dims=['samples', 'clusters'], name="clusters",
                             coords=dict( samples=self._samples, clusters=[0]), attrs=self._attrs )

    def cluster( self, data: xa.DataArray, y=None ):
        self._attrs = data.attrs
        self._samples = data.coords[ data.dims[0] ]
 #       drange = [data.values.min(),data.values.max()]
 #       normed_data: np.ndarray = (data.values - drange[0])/(drange[1]-drange[0])
        self._cluster_data = np.expand_dims( self._model.fit_predict( data.values ), axis=1 ) + 1

    def _update_nclusters( self ):
        self.update_model()






