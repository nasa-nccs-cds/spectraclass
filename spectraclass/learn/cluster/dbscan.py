from spectraclass.util.logs import LogManager, lgm, log_timing
from sklearn import cluster
from joblib import cpu_count
import xarray as xa
import numpy as np
from .base import ClusterBase

class DBScan(ClusterBase):

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
        self._model = cluster.DBSCAN(**params)

    @property
    def cluster_data(self) -> xa.DataArray:
        lgm().log( f"cluster_data: data.shape={self._cluster_data.shape}, samples.shape = {self._samples.shape}")
        return xa.DataArray( self._cluster_data, dims=['samples', 'clusters'], name="clusters",
                             coords=dict( samples=self._samples, clusters=[0]), attrs=self._attrs )

    def cluster( self, data: xa.DataArray, y=None ):
        self._attrs = data.attrs
        self._samples = data.coords[ data.dims[0] ]
        self._cluster_data = np.expand_dims( self._model.fit_predict( data.values ), axis=1 ) + 1

    def _update_nclusters( self ):
        self.update_model()






