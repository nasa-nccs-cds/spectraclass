import xarray
from sklearn import cluster
from sklearn.base import ClusterMixin
from joblib import cpu_count
import time, traceback, shutil
import xarray as xa
import ipywidgets as ipw
from functools import partial
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import traitlets as tl
import traitlets.config as tlc
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable

def clm() -> "ClusterManager":
    return ClusterManager.instance()

class ClusterManager(SCSingletonConfigurable):
    modelid = tl.Unicode("kmeans").tag(config=True, sync=True)
    nclusters = tl.Int(5).tag(config=True, sync=True)
    random_state = tl.Int(0).tag(config=True, sync=True)

    def __init__(self,  **kwargs ):
        super(ClusterManager, self).__init__(**kwargs)
        self._ncluster_options = range( 2, 20 )
        self._mid_options = [ "kmeans" ]
        self.labels_raster: xa.DataArray = None
        self._models: Dict[str,ClusterMixin] = {}
        self._model_selector = ipw.Select( options=self.mids, description='Methods:', value=self.modelid, disabled=False,
                                          layout=ipw.Layout(width="500px"))
        self._ncluster_selector = ipw.Select( options=self._ncluster_options, description='#Clusters:', disabled=False,
                                             value=self.nclusters, layout=ipw.Layout(width="500px"))
        self.update_model()

    def update_model(self):
        self._models[ self.mid ] = self.create_model( self.mid )

    @property
    def mids(self) -> List[str]:
        return self._mid_options

    def create_model(self, mid: str ) -> ClusterMixin:
        if mid == "kmeans":
            params = dict( n_clusters= self._ncluster_selector.value,
                           random_state= self.random_state,
                           batch_size= 256 * cpu_count() )
            return cluster.MiniBatchKMeans( **params )

    def on_parameter_change(self, *args ):
        self.update_model()

    @property
    def mid(self) -> str:
        return self._model_selector.value

    @property
    def model(self) -> ClusterMixin:
        return self._models[ self.mid ]

    def cluster(self, data: xa.DataArray ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        lgm().log( f"Creating clusters from input data shape = {data.shape}")
        block = tm().getBlock()
        samples = data.dims[0]
        cluster_labels = np.expand_dims( self.model.fit_predict( data.values ), axis=1 )
        xa_cluster_samples = xa.DataArray( cluster_labels, dims=[samples,'clusters'], name=f"{block.data.name}_clusters",
                                           coords={samples:data.coords[samples],'clusters':[0]}, attrs=data.attrs )
        self.labels_raster = block.points2raster( xa_cluster_samples ).squeeze()
        return self.labels_raster

        # nodata_value = -2
        # template = self.block.data[0].squeeze(drop=True)
        # self.label_map: xa.DataArray = xa.full_like(template, 0, dtype=np.dtype(
        #     np.int32))  # .where( template.notnull(), nodata_value )
        # #        self.label_map.attrs['_FillValue'] = nodata_value
        # self.label_map.name = f"{self.block.data.name}_labels"
        # self.label_map.attrs['long_name'] = "labels"
        # self.cspecs = lm().get_labels_colormap()
        # lgm().log(f"Init label map, value range = [{self.label_map.values.min()} {self.label_map.values.max()}]")
        # self.labels_image = self.label_map.plot.imshow(ax=self.base.gax, alpha=self.layers('labels').visibility,
        #                                                cmap=self.cspecs['cmap'], add_colorbar=False,
        #                                                norm=self.cspecs['norm'])

    @exception_handled
    def gui(self) -> ipw.DOMWidget:
        selectors = [ self._model_selector,self._ncluster_selector ]
        for selector in selectors: selector.observe( self.on_parameter_change, names=['value'] )
        return ipw.HBox(selectors, layout=ipw.Layout(width="600px", height="300px", border='2px solid firebrick'))
