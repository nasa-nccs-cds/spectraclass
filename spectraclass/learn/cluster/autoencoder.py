from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from spectraclass.reduction.embedding import ReductionManager, rm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import numpy as np
import xarray as xa
from .base import ClusterBase

class AutoEncoderCluster(ClusterBase):

    def __init__( self, n_clusters: int, **kwargs ):
        ClusterBase.__init__( self, n_clusters )
        self.nepochs = kwargs.get( 'nepochs', 10 )
        self.method = kwargs.get('method', "autoencoder" )
        self.cscale = np.full( [1,self._n_clusters], 0.5 )
        self._reduction = None
        self._input_data = None

    @property
    def cluster_data(self) -> np.ndarray:
        lgm().log( f" CLUSTER-AutoEncoder: reduction.shape={self._reduction.shape}, cscale.shape={self.cscale.shape}" )
        return np.argmax(self._reduction * self.cscale, axis=1, keepdims=True )

    def compute_reduction(self):
        ( self._reduction, reproduction, point_data ) = rm().reduce( self._input_data, None, self.method, self._n_clusters, self.nepochs)[0]

    def cluster( self,  data: xa.DataArray, y=None ) -> xa.DataArray:
        self._input_data = data
        self.compute_reduction()
        samples = data.dims[0]
        return xa.DataArray( self.cluster_data, dims=[samples, 'clusters'], name="clusters",
                             coords={samples: data.coords[samples], 'clusters': [0]}, attrs=data.attrs)

    def rescale(self, index: int, sval: float ) -> np.ndarray:
        self.cscale[ index ] = sval
        return self.cluster_data

    def _update_nclusters( self ):
        self.compute_reduction()
        self.cscale = np.full( [1, self._n_clusters], 0.5 )




