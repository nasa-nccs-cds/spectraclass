from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from spectraclass.reduction.embedding import ReductionManager, rm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import numpy as np
import xarray as xa
from .base import ClusterBase

class AutoEncoderCluster(ClusterBase):

    def __init__( self, n_clusters: int, **kwargs ):
        ClusterBase.__init__( self, n_clusters )
        self.nepochs = kwargs.get( 'nepochs',   10 )
        self.method = kwargs.get('method', "autoencoder" )
        self._reduction = None
        self._input_data = None

    @property
    def cluster_data(self) -> xa.DataArray:
        lgm().log( f" CLUSTER-AutoEncoder: reduction.shape={self._reduction.shape}, cscale={self.cscale}" )
        cdata = np.expand_dims( np.argmax(self._reduction * self.cscale, axis=1 ), 1 )
        return xa.DataArray( cdata, dims=['samples', 'clusters'], name="clusters", coords=dict(samples=self._samples, clusters=[0]), attrs=self._attrs )

    def compute_reduction(self):
        ( self._reduction, reproduction, point_data ) = rm().reduce( self._input_data, None, self.method, self._n_clusters, self.nepochs)[0]

    def cluster( self,  data: xa.DataArray, y=None ) -> xa.DataArray:
        self._input_data = data
        self._attrs = data.attrs
        self._samples = data.coords[ data.dims[0] ]
        self.compute_reduction()
        return self.cluster_data

    def _update_nclusters( self ):
        self.compute_reduction()




