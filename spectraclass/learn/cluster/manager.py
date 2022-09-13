import pickle

from joblib import cpu_count
from spectraclass.gui.spatial.widgets.markers import Marker
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.backend_bases import MouseEvent
import xarray as xa
import ipywidgets as ipw
from functools import partial
import numpy as np
from typing import List, Tuple, Dict
import traitlets as tl
from spectraclass.data.spatial.tile.tile import Block
from .base import ClusterBase
from spectraclass.util.logs import lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable

def clm() -> "ClusterManager":
    return ClusterManager.instance()

class ClusterMagnitudeWidget(ipw.HBox):

    def __init__(self, index: int, **kwargs ):
        color = f"rgb{ clm().cluster_color(index) }"
        self.init_value = kwargs.get( 'value', 1.0 )
        handler = kwargs.get( 'handler', False )
        range = kwargs.get( 'range', [0.0,4.0] )
        cname = "Threshold" if index == 0 else f"Cluster-{index}"
        self.label = ipw.Button( description=cname, style=dict( button_color=color ) )
        self._index = index
        self.slider = ipw.FloatSlider( self.init_value, description="", min=range[0], max=range[1], step=0.1 )
        self.label.on_click( self.reset )
        ipw.HBox.__init__( self, [self.label,self.slider] )
        if handler: self.on_change( handler )

    def on_change(self, handler ):
        self.slider.observe( partial( handler, self._index ), 'value' )

    def reset(self, *args ):
        self.slider.value = self.init_value

class ClusterManager(SCSingletonConfigurable):
    modelid = tl.Unicode("kmeans").tag(config=True, sync=True)
    nclusters = tl.Int(5).tag(config=True, sync=True)
    random_state = tl.Int(0).tag(config=True, sync=True)

    def __init__(self, **kwargs ):
        super(ClusterManager, self).__init__(**kwargs)
        self._max_culsters = 15
        self._ncluster_options = range( 2, self._max_culsters )
        self._mid_options = [ "kmeans", "fuzzy cmeans", "autoencoder" ] # , "umap" ] # "hierarchical" "DBSCAN", "spectral" ]
        self._cluster_colors: np.ndarray = None
        self._cluster_raster: xa.DataArray = None
        self._marked_colors: Dict[Tuple,Tuple[float,float,float]] = {}
        self._marked_clusters: Dict[Tuple, List] = {}
        self.thresh_slider = None
        self._cluster_points: xa.DataArray = None
        self._models: Dict[str,ClusterBase] = {}
        self._model_selector = ipw.Select( options=self.mids, description='Methods:', value=self.modelid, disabled=False,
                                          layout=ipw.Layout(width="500px"))
        self._ncluster_selector = ipw.Select( options=self._ncluster_options, description='#Clusters:', disabled=False,
                                             value=self.nclusters, layout=ipw.Layout(width="500px"))
        self.update_model()

    def update_model(self):
        self._models[ self.mid ] = self.create_model( self.mid )

    def add_model(self, mid: str, clusterer: ClusterBase ):
        self._models[ mid ] = clusterer

    def update_colors(self, ncolors: int):
        hsv = np.full( [ncolors,3], 1.0 )
        hsv[:,0] = np.linspace( 0.0, 1.0, ncolors+1 )[:ncolors]
        hsv[:, 1] = np.full( [ncolors], 0.4 )
        self._cluster_colors = np.full( [ncolors+1,3], 1.0 )
        self._cluster_colors[1:ncolors+1,:] = hsv_to_rgb(hsv)

    @property
    def mids(self) -> List[str]:
        return self._mid_options

    def create_model(self, mid: str ) -> ClusterBase:
        from .fcm import FCM
        from  .kmeans import KMeansCluster
        from .dbscan import DBScan
        from .spectral import Spectral
        nclusters = self._ncluster_selector.value
        self.update_colors( self._max_culsters )
        lgm().log( f"Creating {mid} model with {nclusters} clusters")
        if mid == "kmeans":
            params = dict(  random_state= self.random_state, batch_size= 256 * cpu_count() )
            return KMeansCluster( nclusters, **params )
        if mid == "fuzzy cmeans":
            return FCM( nclusters )
        elif mid == "dbscan":
             eps = 0.001 / nclusters
             return DBScan( eps=eps, min_samples=10, n_clusters=nclusters )
        elif mid == "spectral":
            return Spectral( n_clusters=nclusters )

        # elif mid == "hierarchical":
        #     return cluster.AgglomerativeClustering( linkage="ward", n_clusters=nclusters ) # , connectivity= )


    def on_parameter_change(self, *args ):
        self.update_model()

    @property
    def mid(self) -> str:
        return self._model_selector.value

    @property
    def model(self) -> ClusterBase:
        return self._models[ self.mid ]

    def get_colormap( self, layer: bool ):
        return self.get_layer_colormap() if layer else self.get_cluster_colormap()

    def get_cluster_colors( self ) ->  np.ndarray:
        colors: np.ndarray = self._cluster_colors.copy()
        for (icluster, value) in self.marked_colors.items(): colors[icluster] = value
        return colors

    def get_cluster_colormap( self ) -> LinearSegmentedColormap:
        colors: np.ndarray = self.get_cluster_colors()
        lgm().log( f'get_cluster_colormap: ncolors={colors.shape[0]}, colors = {colors.tolist()}')
        return LinearSegmentedColormap.from_list( 'clusters', colors, N=colors.shape[0] )

    @property
    def marked_colors(self) -> Dict[int,Tuple[float,float,float]]:
        mcolors = {}
        for (ckey, value) in self._marked_colors.items():
            icluster = self.get_icluster( ckey )
            if icluster >= 0: mcolors[icluster] = value
        return mcolors

    def get_icluster( self, ckey: Tuple ) -> int:
        from spectraclass.data.spatial.tile.manager import tm
        ( tindex, bindex, icluster ) = ckey
        return icluster if ( (tindex==tm().image_index) and (bindex==tm().block_coords) )  else -1

    def cluster_color(self, index: int ) -> Tuple[int,int,int]:
        if index == 0: return ( 1, 1, 1 )
        else:
            colors = self.get_cluster_colors()
            rgb: np.ndarray = colors[ index ] * 255.99
            return tuple( rgb.astype(np.int).tolist() )

    def get_layer_colormap( self ):
        ncolors = self._cluster_colors.shape[0]
        colors = np.full( [ncolors,4], 0.0 )
        for (key, value) in self.marked_colors.items(): colors[key] = list(value) + [1.0]
        return LinearSegmentedColormap.from_list( 'cluster-layer', colors, N=ncolors )

    def clear(self, reset=True ):
        self._cluster_points = None
        self._cluster_raster = None
        self.threshold_mask = None
        if self.thresh_slider is not None:
            self.thresh_slider.value = 0.0
        if reset: self.model.reset()

    def run_cluster_model( self, data: xa.DataArray ):
        lgm().log( f"Creating clusters from input data shape = {data.shape}")
        self.clear()
        self.model.cluster(data)
        self._cluster_points = self.model.cluster_data

    def cluster(self, data: xa.DataArray ) -> xa.DataArray:
        self.run_cluster_model( data )
        return self.get_cluster_map()

    @exception_handled
    def get_cluster_map( self, layer: bool = False ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import tm
        if self._cluster_raster is None:
            block = tm().getBlock()
            self._cluster_raster: xa.DataArray = block.points2raster( self._cluster_points ).squeeze()
        self._cluster_raster.attrs['cmap'] = self.get_colormap( layer )
        return self._cluster_raster

    @property
    def samples(self) -> np.ndarray:
        return self._cluster_points.samples.values

    def gid2pid( self, gid: int ) -> int:
        pids = np.where( self.samples == gid )
        if isinstance(pids,tuple): pids = pids[0]
        return pids[0] if len(pids) else -1

    def get_cluster(self, gid: int ) -> int:
        pid: int = self.gid2pid( gid )
        if pid >= 0:
            result = self._cluster_points.values[pid, 0]
            return result
        else:
            lgm().log( f" ------> Can find cluster: gid={gid}, samples: gid-in={gid in self.samples}, size={self.samples.size}, range={[self.samples.min(),self.samples.max()]}")
            pickle.dump( self.samples.tolist(), open("/tmp/cluster_gids.pkl","wb") )
            return -1

    def get_marked_clusters( self, cid: int ) -> List[int]:
        from spectraclass.data.spatial.tile.manager import tm
        ckey = ( tm().image_index, tm().block_coords, cid )
        return self._marked_clusters.setdefault( ckey, [] )

    def get_points(self, cid: int ) -> np.ndarray:
        class_points = np.array( [], dtype=np.int )
        clusters: List[int] = self.get_marked_clusters(cid)
        for icluster in clusters:
            mask = ( self._cluster_points.values.squeeze() == icluster )
            pids: np.ndarray = self._cluster_points.samples[mask].values
            class_points = np.concatenate( (class_points, pids), axis=0 )
        return class_points.astype(np.int)

    @property
    def cluster_points(self) -> xa.DataArray:
        return self._cluster_points

    @log_timing
    def mark_cluster( self, gid: int, cid: int, icluster: int ) -> Marker:
        from spectraclass.model.labels import lm
        from spectraclass.data.spatial.tile.manager import tm
        ckey = ( tm().image_index, tm().block_coords, icluster )
        self._marked_colors[ ckey ] = lm().get_rgb_color(cid)
        self.get_marked_clusters(cid).append( icluster )
        cmap = self.get_cluster_map().values
        marker = Marker("clusters", self.get_points(cid), cid, mask=(cmap == icluster))
        lgm().log(f"#IA: mark_cluster[{icluster}]: ckey={ckey} cid={cid}, #pids = {marker.size}")
        return marker

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
        return ipw.VBox( [ ipw.HBox( selectors, layout=ipw.Layout(width="600px", height="200px", border='2px solid firebrick') ), self.tuning_gui() ] )

    @exception_handled
    def tuning_gui(self) -> ipw.DOMWidget:
        self.thresh_slider = ClusterMagnitudeWidget( 0, range=[0.0,1.0], value=0.0, handler=self.tune_cluster )
        tuning_sliders= [ self.thresh_slider ]
        for icluster in range( 1, self._max_culsters+1 ):
            tuning_sliders.append( ClusterMagnitudeWidget( icluster, handler=self.tune_cluster ) )
        return  ipw.VBox( tuning_sliders, layout=ipw.Layout( width="600px", border='2px solid firebrick' ) )

    @exception_handled
    def tune_cluster(self, icluster: int, change: Dict ):
        from spectraclass.gui.spatial.map import mm
        self.clear( reset=False )
        self.model.rescale(icluster, change['new'])
        self._cluster_points = self.model.cluster_data
        mm().plot_cluster_image( self.get_cluster_map() )

class ClusterSelector:
    LEFT_BUTTON = 1

    def __init__(self, ax ):
        self.ax = ax
        self.enabled = False
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('button_press_event', self.on_button_press)

    @property
    def block(self) -> Block:
        from spectraclass.data.spatial.tile.manager import tm
        return tm().getBlock()

    def set_enabled(self, enable: bool ):
        lgm().log( f"ClusterSelector: set enabled = {enable}")
        self.enabled = enable

    def clear(self):
        clm().clear()

    @exception_handled
    def on_button_press(self, event: MouseEvent ):
        from spectraclass.gui.spatial.map import mm
        from spectraclass.gui.spatial.widgets.markers import Marker
        from spectraclass.application.controller import app
        from spectraclass.model.labels import lm
        from spectraclass.gui.control import ufm
        lgm().log(f"ClusterSelector: on_button_press: enabled={self.enabled}")
        if (event.xdata != None) and (event.ydata != None) and (event.inaxes == self.ax) and self.enabled:
            if int(event.button) == self.LEFT_BUTTON:
                gid,ix,iy = self.block.coords2gid(event.ydata, event.xdata)
                cid = lm().current_cid
                icluster = clm().get_cluster(gid)
                ufm().show(f"Mark cluster: ({ix},{iy})-> {gid}: cluster = {icluster}", color="blue")
                lgm().log(f"#IA: mark_cluster: [{ix},{iy}]->{gid}, cid={cid}, cluster = {icluster}")
                if icluster >= 0:
                    marker: Marker = clm().mark_cluster(gid, cid, icluster)
                    app().add_marker( "cluster", marker )
                    mm().plot_cluster_image( clm().get_cluster_map() )
#                    labels_image: xa.DataArray = lm().get_label_map()
#                    mm().plot_markers_image()
#                    lm().addAction( "cluster", "application", cid=cid )