import xarray
from sklearn import cluster
from sklearn.base import ClusterMixin
from joblib import cpu_count
import time, traceback, shutil
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, hsv_to_rgb
from matplotlib.backend_bases import PickEvent, MouseEvent
import xarray as xa
import ipywidgets as ipw
from functools import partial
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import traitlets as tl
from spectraclass.data.spatial.tile.tile import Block, Tile
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

    def __init__(self, **kwargs ):
        super(ClusterManager, self).__init__(**kwargs)
        self._ncluster_options = range( 2, 20 )
        self._mid_options = [ "kmeans" ]
        self._cluster_colors: np.ndarray = None
        self._cluster_raster: xa.DataArray = None
        self._marked_colors: Dict[int,Tuple[float,float,float]] = {}
        self._markers = {}
        self._cluster_points: xa.DataArray = None
        self._models: Dict[str,ClusterMixin] = {}
        self._model_selector = ipw.Select( options=self.mids, description='Methods:', value=self.modelid, disabled=False,
                                          layout=ipw.Layout(width="500px"))
        self._ncluster_selector = ipw.Select( options=self._ncluster_options, description='#Clusters:', disabled=False,
                                             value=self.nclusters, layout=ipw.Layout(width="500px"))
        self.update_model()

    def update_model(self):
        self._models[ self.mid ] = self.create_model( self.mid )

    def update_colors(self, ncolors: int):
        hsv = np.full( [ncolors,3], 1.0 )
        hsv[:,0] = np.linspace( 0.0, 1.0, ncolors+1 )[:ncolors]
#        hsv[:,1] = np.array( [ 1.0-0.5*(i%2) for i in range(ncolors) ] )
        hsv[:, 1] = np.full( [ncolors], 0.5 )
        self._cluster_colors = hsv_to_rgb(hsv)
        lgm().log( f"UPDATE COLORS[{ncolors}], colormap shape = {self._cluster_colors.shape}")

    @property
    def mids(self) -> List[str]:
        return self._mid_options

    def create_model(self, mid: str ) -> ClusterMixin:
        nclusters = self._ncluster_selector.value
        self.update_colors( nclusters )
        lgm().log( f"Creating {mid} model with {nclusters} clusters")
        if mid == "kmeans":
            params = dict( n_clusters= nclusters,
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

    def get_colormap( self, layer: bool ):
        return self.get_layer_colormap() if layer else self.get_cluster_colormap()

    def get_cluster_colormap( self ):
        colors = self._cluster_colors.copy()
        for (key, value) in self._marked_colors.items(): colors[key] = value
        return LinearSegmentedColormap.from_list( 'clusters', colors, N=len(colors) )

    def get_layer_colormap( self ):
        ncolors = self._cluster_colors.shape[0]
        colors = np.full( [ncolors,4], 0.0 )
        for (key, value) in self._marked_colors.items(): colors[key] = list(value) + [1.0]
        return LinearSegmentedColormap.from_list( 'cluster-layer', colors, N=ncolors )

    def run_cluster_model( self, data: xa.DataArray ):
        lgm().log( f"Creating clusters from input data shape = {data.shape}")
        samples = data.dims[0]
        cluster_data = np.expand_dims( self.model.fit_predict( data.values ), axis=1 )
        self._cluster_points = xa.DataArray( cluster_data, dims=[samples,'clusters'],  name="clusters",
                                           coords={samples:data.coords[samples],'clusters':[0]}, attrs=data.attrs )
        self._cluster_raster = None

    def cluster(self, data: xa.DataArray ) -> xa.DataArray:
        self.run_cluster_model( data )
        return self.get_cluster_map()

    def get_cluster_map( self, layer: bool = False ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if self._cluster_raster is None:
            block = tm().getBlock()
            self._cluster_raster: xa.DataArray = block.points2raster( self._cluster_points ).squeeze()
        self._cluster_raster.attrs['cmap'] = self.get_colormap( layer )
        return self._cluster_raster

    def get_cluster(self, pid: int ) -> int:
        clusters = self._cluster_points.values.squeeze()
        return clusters[pid]

    def get_points(self, cid: int ) -> np.ndarray:
        class_points = np.array( [], dtype=np.int )
        classes: List[int] = self._markers.get( cid, [] )
        for iclass in classes:
            mask = ( self._cluster_points.values.squeeze() == iclass )
            pids: np.ndarray = self._cluster_points.samples[mask].values
            class_points = np.concatenate( (class_points, pids), axis=0 )
        return class_points.astype(np.int)

    def mark_cluster( self, pid: int, cid: int ) -> xa.DataArray:
        from spectraclass.model.labels import LabelsManager, lm
        iClass = self.get_cluster( pid )
        lgm().log( f"Mark cluster, pid={pid}, iClass={iClass}, cid={cid}")
        ufm().show( f"Label cluster, cluster[{iClass}] -> class[{cid}]" )
        self._marked_colors[ iClass ] = lm().get_rgb_color(cid)
        self._markers.setdefault( cid, [] ).append( iClass )
        return self.get_cluster_map()

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

class ClusterSelector:
    LEFT_BUTTON = 1

    def __init__(self, ax, block: Block ):
        self.ax = ax
        self.enabled = False
        self.block: Block = block
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('button_press_event', self.on_button_press)

    def set_enabled(self, enable: bool ):
        lgm().log( f"ClusterSelector: set enabled = {enable}")
        self.enabled = enable

    @exception_handled
    def on_button_press(self, event: MouseEvent ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.gui.spatial.widgets.markers import Marker
        from spectraclass.application.controller import app
        from spectraclass.model.labels import LabelsManager, lm
        lgm().log(f"ClusterSelector: on_button_press: enabled={self.enabled}")
        if (event.xdata != None) and (event.ydata != None) and (event.inaxes == self.ax) and self.enabled:
            if int(event.button) == self.LEFT_BUTTON:
                pid = self.block.coords2pindex(event.ydata, event.xdata)
                if pid >= 0:
                    cid = lm().current_cid
                    t0 = time.time()
                    cluster_map: xa.DataArray = clm().mark_cluster( pid, cid )
                    marker = Marker( "labels", clm().get_points(cid), cid )
                    t1 = time.time()
                    app().add_marker( "map", marker )
                    t2 = time.time()
                    mm().plot_cluster_image( cluster_map )
                    t3 = time.time()
                    labels_image: xa.DataArray = lm().get_label_map()
                    t4 = time.time()
                    mm().plot_labels_image( labels_image )
                    t5 = time.time()
                    lgm().log( f"CLUSTER PLOT: {t1-t0:.2f} {t2-t1:.2f} {t3-t2:.2f} {t4-t3:.2f} {t5-t4:.2f} ")
#                    lm().addAction( "cluster", "application", cid=cid )