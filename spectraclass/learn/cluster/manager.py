import pickle, math
from random import random
import panel as pn
from panel.layout import Panel
import holoviews as hv
from panel.widgets import Button, Select, FloatSlider
from joblib import cpu_count
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.gui.control import UserFeedbackManager, ufm
from holoviews.streams import SingleTap, DoubleTap
import xarray as xa
from functools import partial
import numpy as np
from typing import List, Tuple, Dict
from param.parameterized import Event
import traitlets as tl
from spectraclass.data.spatial.tile.tile import Block
from .base import ClusterBase
from spectraclass.util.logs import lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable
import colorsys
from holoviews.streams import Stream, param

Count = Stream.define('Count', index=param.Integer(default=0, doc='Cluster Operation count'))

def arange( data: xa.DataArray, axis=None ) -> Tuple[np.ndarray,np.ndarray]:
    return ( np.nanmin(data.values,axis=axis), np.nanmax(data.values,axis=axis) )

def bounds( raster: xa.DataArray ) -> Tuple[Tuple,Tuple]:
    xc, yc = raster.coords['x'].to_numpy(), raster.coords['y'].to_numpy()
    dx, dy = xc[1]-xc[0], yc[1]-yc[0]
    return ( xc[0]-dx, xc[-1]+dx), ( yc[0]-dy, yc[-1]+dy)

def cindx( v: float ) -> int:
    return math.floor( v*255.99 )

def rgb_to_hex( r, g, b ) -> str:
    return f'#{cindx(r):02x}{cindx(g):02x}{cindx(b):02x}'

def random_hex_color() -> str:
    return rgb_to_hex( random(), random(), random() )

def clm() -> "ClusterManager":
    return ClusterManager.instance()

ThresholdStream = Stream.define( 'ThresholdStream', tindex=0, tvalue=1.0 )

class ClusterMagnitudeWidget:
    height = 26

    def __init__(self, index: int, tstream: Stream, **kwargs ):
      #  cluster_color = f"rgb{ clm().cluster_color(index) }"
        self.init_value = kwargs.get( 'value', 1.0 )
        self.tstream: ThresholdStream = tstream
        range = kwargs.get( 'range', [0.0,2.0] )
        step = kwargs.get( 'step', 0.05 )
        cname = "Threshold" if index == 0 else f"Cluster-{index}"
        self.label = Button( name=cname, button_type='primary' ) # .opts( color=cluster_color ) name='Click me', button_type='primary'
        self._index = index
        self.slider = FloatSlider( name='', start=range[0], end=range[1], step=step, value=self.init_value ) #self.init_value, description="", min=range[0], max=range[1], step=step )
        self.label.on_click( self.reset )
        self.slider.param.watch( self.update, ['value'], onlychanged=True )

    @exception_handled
    def update(self, event: Event ):
        if type(event) == Event:
            tvalue = event.new
            self.tstream.event( tindex=self._index, tvalue=tvalue )
            lgm().log( f"CM: tstream.event {self._index} {tvalue}")

    def panel(self):
        return pn.Row(  self.label, self.slider )  # , layout=ipw.Layout( width="550px", height=f"{self.height}px"), overflow="hidden" )

    def set_color(self, color: str ):
        self.label.apply.opts( color = color )

    def reset_color(self ):
        self.label.apply.opts( color =  f"rgb{ clm().cluster_color( self._index, False ) }" )

    def reset(self, *args ):
        self.slider.value = self.init_value

class ClusterManager(SCSingletonConfigurable):
    modelid = tl.Unicode("kmeans").tag(config=True, sync=True)
    nclusters = tl.Int(5).tag(config=True, sync=True)
    random_state = tl.Int(0).tag(config=True, sync=True)

    def __init__(self, **kwargs ):
        super(ClusterManager, self).__init__(**kwargs)
        self.width = kwargs.pop('width',600)
        self.thresholdStream = ThresholdStream()
        self.double_tap_stream = DoubleTap(transient=True)
        self._max_culsters = 20
        self.cmap = [ random_hex_color() for i in range( 0, self._max_culsters ) ]
        self._ncluster_options = list( range( 2, self._max_culsters ) )
        self._count = Count(index=0)
        self._mid_options = [ "kmeans", "fuzzy cmeans", "bisecting kmeans" ]
        self._cluster_colors: np.ndarray = None
        self._cluster_raster: xa.DataArray = None
        self._cluster_image = hv.DynamicMap( self.get_cluster_image, streams=[ self._count, self.double_tap_stream, self.thresholdStream ] )    # thresholdStream
        self._marked_colors: Dict[Tuple,Tuple[float,float,float]] = {}
        self._marked_clusters: Dict[Tuple, List] = {}
        self._tuning_sliders: List[ClusterMagnitudeWidget] = []
        self.thresh_slider = None
        self._cluster_points: xa.DataArray = None
        self._cluster_markers: Dict[int, Marker] = {}
        self._models: Dict[str,ClusterBase] = {}
        self._model_selector = pn.widgets.Select(name='Methods', options=self.mids, value=self.modelid )
        self._model_watcher = self._model_selector.param.watch( self.on_parameter_change, ['value'], onlychanged=True )
        self._ncluster_selector = pn.widgets.Select(name='#Clusters', options=self._ncluster_options, value=self.nclusters )
        self._ncluster_watcher = self._ncluster_selector.param.watch(self.on_parameter_change, ['value'], onlychanged=True )

    def refresh(self) -> int:
        ccount = self._count.index + 1
        self._count.event( index=ccount )
        return ccount

    def update_model(self):
        if self.mid not in self._models:
            self._models[ self.mid ] = self.create_model( self.mid )

    def add_model(self, mid: str, clusterer: ClusterBase ):
        self._models[ mid ] = clusterer

    def update_colors(self, ncolors: int):
        hsv = np.full( [ncolors,3], 1.0 )
        hsv[:,0] = np.linspace( 0.0, 1.0, ncolors+1 )[:ncolors]
        hsv[:, 1] = np.full( [ncolors], 0.4 )
        self._cluster_colors = np.full( [ncolors+1,3], 1.0 )
        for iC in range( ncolors ):
            self._cluster_colors[iC+1,:] = colorsys.hsv_to_rgb( *hsv[iC].tolist() )

    @property
    def mids(self) -> List[str]:
        return self._mid_options

    def create_model(self, mid: str ) -> ClusterBase:
        from .fcm import FCM
        from  .kmeans import KMeansCluster, BisectingKMeans
        nclusters = self._ncluster_selector.value
        self.update_colors( self._max_culsters )
        lgm().log( f"#CM: Creating {mid} model with {nclusters} clusters")
        if mid == "kmeans":
            params = dict(  random_state= self.random_state, batch_size= 256 * cpu_count() )
            return KMeansCluster( nclusters, **params )
        if mid == "fuzzy cmeans":
            return FCM( nclusters )
        elif mid == "bisecting kmeans":
             return BisectingKMeans( n_clusters=nclusters )

    def on_parameter_change(self, *args ):
        self.update_model()

    @property
    def mid(self) -> str:
        return self._model_selector.value

    @property
    def model(self) -> ClusterBase:
        self.update_model()
        return self._models[ self.mid ]

    def get_cluster_colors( self, updated=True ) ->  np.ndarray:
        colors: np.ndarray = self._cluster_colors.copy()
        if updated:
            for (icluster, value) in self.marked_colors.items(): colors[icluster] = value
        return colors

    # def get_cluster_colormap( self ) -> LinearSegmentedColormap:
    #     colors: np.ndarray = self.get_cluster_colors()
    #     lgm().log( f'get_cluster_colormap: ncolors={colors.shape[0]}, colors = {colors.tolist()}')
    #     return LinearSegmentedColormap.from_list( 'clusters', colors, N=colors.shape[0] )

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

    def cluster_color(self, index: int, updated = True ) -> Tuple[int,int,int]:
        if index == 0: return ( 1, 1, 1 )
        else:
            colors = self.get_cluster_colors(updated)
            rgb: np.ndarray = colors[ index ] * 255.99
            return tuple( rgb.astype(np.int32).tolist() )

    # def get_layer_colormap( self ):
    #     ncolors = self._cluster_colors.shape[0]
    #     colors = np.full( [ncolors,4], 0.0 )
    #     for (key, value) in self.marked_colors.items(): colors[key] = list(value) + [1.0]
    #     return LinearSegmentedColormap.from_list( 'cluster-layer', colors, N=ncolors )

    def clear(self, reset=True ):
        self._cluster_points = None
        self._cluster_raster = None
        self.threshold_mask = None
        if self.thresh_slider is not None:
            self.thresh_slider.value = 0.0
        if reset: self.model.reset()

    def run_cluster_model( self, data: xa.DataArray ):
        self.nclusters = self._ncluster_selector.value
        self.clear()
        self.model.n_clusters = self.nclusters
        lgm().log( f"#CM: Creating {self.nclusters} clusters from input data shape = {data.shape}")
        self.model.cluster(data)
        self._cluster_points = self.model.cluster_data

    @exception_handled
    def cluster(self, data: xa.DataArray ):
        self.run_cluster_model( data )
        ccount = self.refresh()
        lgm().log( f"#CM: exec cluster, op count={ccount}" )

    @exception_handled
    def get_cluster_map( self ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import tm
        from spectraclass.data.base import DataManager, dm
        block = tm().getBlock()
        if self.cluster_points is None:
            self.cluster( dm().getModelData(block=block) )
        self._cluster_raster: xa.DataArray = block.points2raster( self.cluster_points, name="Cluster" ).squeeze()
        self._cluster_raster.attrs['title'] = f"Block = {block.block_coords}"
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
            lgm().log( f"#CM: Can find cluster: gid={gid}, samples: gid-in={gid in self.samples}, size={self.samples.size}, range={[self.samples.min(),self.samples.max()]}")
            pickle.dump( self.samples.tolist(), open("/tmp/cluster_gids.pkl","wb") )
            return -1

    def get_marked_clusters( self, cid: int ) -> List[int]:
        from spectraclass.data.spatial.tile.manager import tm
        ckey = ( tm().image_index, tm().block_coords, cid )
        return self._marked_clusters.setdefault( ckey, [] )

    def get_points(self, cid: int ) -> np.ndarray:
        class_points = np.array( [], dtype=np.int32 )
        clusters: List[int] = self.get_marked_clusters(cid)
        for icluster in clusters:
            mask = ( self._cluster_points.values.squeeze() == icluster )
            pids: np.ndarray = self._cluster_points.samples[mask].values
            class_points = np.concatenate( (class_points, pids), axis=0 )
        return class_points.astype(np.int32)

    def get_cluster_pids(self, icluster: int ) -> np.ndarray:
        mask = ( self._cluster_points.values.squeeze() == icluster )
        return self._cluster_points.samples[mask].values

    @property
    def cluster_points(self) -> xa.DataArray:
        return self._cluster_points

    @exception_handled
    def mark_cluster( self, cid: int, icluster: int ) -> Marker:
        from spectraclass.model.labels import lm
        from spectraclass.data.spatial.tile.manager import tm
        ckey = ( tm().image_index, tm().block_coords, icluster )
        class_color = lm().get_rgb_color(cid)
        self._marked_colors[ ckey ] = class_color
        cluster_color = rgb_to_hex( *class_color )
        self.cmap[ icluster ] = cluster_color
     #   self._tuning_sliders[ icluster ].set_color( lm().current_color )
        self.get_marked_clusters(cid).append( icluster )
        cmap = self.get_cluster_map().values
        marker = Marker("clusters", self.get_points(cid), cid, mask=(cmap == icluster))
        lgm().log(f"#CM: mark_cluster[{icluster}]: ckey={ckey} cid={cid}, #pids = {marker.size}, cluster_color={cluster_color}")
        self._cluster_markers[icluster] = marker
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
    def panel(self, **kwargs ) -> hv.DynamicMap:
        width = kwargs.get('width', 600)
        height = kwargs.get('height', 500)
        return self._cluster_image.opts( width=width, height=height )

    @exception_handled
    def get_cluster_image( self, index: int, tindex: int, tvalue: int, x=None, y=None ) -> hv.Image:
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.data.spatial.tile.manager import TileManager, tm

        if x is not None:
            block: Block = tm().getBlock()
            gid, ix, iy = block.coords2gid(y,x)
            cid = lm().current_cid
            icluster = clm().get_cluster(gid)
            self.mark_cluster( cid, icluster )
            ufm().show(f"get_cluster_image:  x={x}, y={y}, label='{lm().selectedLabel}'{cid}), icluster={icluster}")
        else: ufm().show(f"get_cluster_image")

        lgm().log( f"#CM: create cluster image[{index}], tindex={tindex}, tvalue={tvalue}, x={x}, y={y}, cmap={self.cmap[:5]}" )
#        self.rescale( tindex, tvalue )
        raster: xa.DataArray = self.get_cluster_map()
        iopts = dict( width=self.width, xaxis="bare", yaxis="bare", x="x", y="y", colorbar=False, title=raster.attrs['title'] )
        image =  raster.hvplot.image( **iopts )
#        xlim, ylim = bounds( raster )
 #       image =  hv.Image( raster.to_numpy(), xlim=xlim, ylim=ylim, colorbar=False, title=raster.attrs['title'], xaxis="bare", yaxis="bare" ).opts(cmap=self.cmap)
        cmaps = ['gray','PiYG','flag','Set1']
        return image.opts( cmap=cmaps[index%4] )  # self.cmap

    @exception_handled
    def gui(self) -> Panel:
        from spectraclass.model.labels import LabelsManager, lm
        selectors = [ self._model_selector,self._ncluster_selector ]
        selection_gui = pn.Row( *selectors )
        actions_panel = pn.Row( *self.action_buttons() )
        selection_controls = pn.WidgetBox( "### Clustering", selection_gui, actions_panel )
        labeling_controls = pn.WidgetBox( "### Labeling", lm().class_selector )
        controls_panel = pn.Column( selection_controls, labeling_controls )
        return pn.Tabs( ("controls",controls_panel), ("tuning",self.tuning_gui()) )

    def action_buttons(self):
        buttons = []
        for task in [ "cluster" ]:  #, "embed" ]:
            button = Button( name=task, button_type='primary' ) # .opts(color=cluster_color) name='Click me', button_type='primary'
            button.on_click( partial( self.on_action, task ) )
            buttons.append( button )
        return buttons

    @exception_handled
    def on_action(self, action: str, *args, **kwargs ):
        if action == "embed":
            self.create_embedding()
        elif action == "cluster":
            self.generate_clusters()

    def generate_clusters(self):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block: Block = tm().getBlock()
        mdata: xa.DataArray = dm().getModelData( block=block )
        ufm().show(f"Creating clusters using {self.mid} for block {block.block_coords}")
        self.cluster( mdata )
        self.refresh()

    def create_embedding(self, ndim: int = 3):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        ufm().show( f"Creating embedding with method '{self.mid}'")
        embedding: xa.DataArray = self.model.embedding(ndim)
        pcm().update_plot( points=embedding )

    def reset_clusters(self):
 #       from spectraclass.application.controller import app
        self._marked_colors = {}
        self._cluster_points = None
        # for (icluster, marker) in self._cluster_markers.items():
        #     if marker.active():
        #         app().remove_marker(marker)
        #         self.get_marked_clusters(marker.cid).remove(icluster)
        # for slider in self._tuning_sliders:
        #     slider.reset_color()

    @exception_handled
    def tuning_gui(self) -> Panel:
        if not self._tuning_sliders:
            self.thresh_slider = ClusterMagnitudeWidget( 0, self.thresholdStream, range=[0.0,1.0], value=0.0, step=0.02 )
            self._tuning_sliders= [ self.thresh_slider ]
            for icluster in range( 1, self._max_culsters+1 ):
                cmw = ClusterMagnitudeWidget( icluster, self.thresholdStream, range=[0.0,2.0], value=1.0, step=0.02 )
                self._tuning_sliders.append( cmw )
        panels = [ ts.panel() for ts in self._tuning_sliders ]
        return  pn.Column( *panels )

    @property
    def max_clusters(self):
        return self._max_culsters + 1

    @exception_handled
    def tune_cluster(self, icluster: int, change: Dict ):
        lgm().log(f"CM: tune_cluster[{icluster}]: change = {change}")
 #       from spectraclass.gui.spatial.map import mm
        self.rescale( icluster, change['new'] )
 #       mm().plot_cluster_image( self.get_cluster_map() )

    @exception_handled
    def rescale(self, icluster: int, value: float ):
        self.clear( reset=False )
        lgm().log( f"CM: rescale cluster-{icluster}: value = {value}")
        self.model.rescale( icluster, value )
        self._cluster_points = None
        # self._cluster_points = self.model.cluster_data
        # if self._cluster_points is not None:
        #     if icluster == 0:
        #         for iC in self._cluster_markers.keys(): self.update_cluster( iC )
        #     else: self.update_cluster(icluster)

    def update_cluster(self, icluster: int ):
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        lgm().log(f"#IA: update_cluster: marked-cids = {list(self._cluster_markers.keys())}")
        marker: Marker = self._cluster_markers.get(icluster,None)
        if marker is not None:
            gpm().remove_marker( marker )
            pcm().deleteMarkers(marker.gids.tolist())
            pids = self.get_cluster_pids( icluster )
            marker.set_gids(pids)
            gpm().plot_graph(marker)
            pcm().addMarker( marker )
            lgm().log( f"#IA: update_cluster, npids={len(pids)}, cluster points shape = {self._cluster_points.shape}")


class ClusterSelector:
    LEFT_BUTTON = 1

    def __init__(self):
        self.enabled = False

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
    def on_button_press(self, event ):
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
                    app().add_marker( marker )
      #              mm().plot_cluster_image( clm().get_cluster_map() )
#                    labels_image: xa.DataArray = lm().get_label_map()
#                    mm().plot_markers_image()
#                    lm().addAction( "cluster", "application", cid=cid )