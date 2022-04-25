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
        self.update_model()

    def update_model(self):
        params = dict( n_clusters=self._n_clusters, random_state=self.random_state, batch_size=self.batch_size )
        self._model = cluster.MiniBatchKMeans(**params)

    def cluster( self, data: xa.DataArray, y=None ) -> xa.DataArray:
        cluster_data = np.expand_dims( self._model.fit_predict( data.values ), axis=1 )
        samples = data.dims[0]
        return xa.DataArray( cluster_data, dims=[samples, 'clusters'], name="clusters",
                             coords={samples: data.coords[samples], 'clusters': [0]}, attrs=data.attrs)

    def _update_nclusters( self ):
        self.update_model()






