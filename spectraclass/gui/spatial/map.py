import xarray as xa
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import logging, os
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.backend_bases import NavigationToolbar2, _Mode
from matplotlib.backend_bases import PickEvent, MouseButton  # , NavigationToolbar2
from functools import partial
from spectraclass.gui.spatial.widgets.markers import MarkerManager
from matplotlib.image import AxesImage
from spectraclass.gui.control import UserFeedbackManager, ufm
from matplotlib.axes import Axes
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.gui.spatial.basemap import TileServiceBasemap
from spectraclass.widgets.polygons import PolygonInteractor
import matplotlib.pyplot as plt
import ipywidgets as ipw
import math, atexit, os, traceback, time
from collections import OrderedDict
from spectraclass.gui.spatial.widgets.layers import LayersManager, Layer
from spectraclass.model.labels import LabelsManager, lm
from matplotlib.image import AxesImage
from spectraclass.xext.xgeo import XGeo
from spectraclass.learn.cluster.manager import ClusterSelector
# from spectraclass.widgets.slider import PageSlider
from  matplotlib.widgets import Slider
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.data.spatial.tile.tile import Block, Tile, ThresholdRecord

def mm(**kwargs) -> "MapManager":
    return MapManager.instance(**kwargs)

def fs(flist):
    return [f"{fv:.1f}" for fv in flist]

def svalid( data: np.ndarray):
    return f"#valid: {np.count_nonzero(~np.isnan(data))}/{data.size}"

srng = lambda x: f"({x.min()},{x.max()})"

class MapManager(SCSingletonConfigurable):
    init_band = tl.Int(10).tag(config=True, sync=True)
    upper_threshold = tl.Float(1.0).tag(config=True, sync=True)
    lower_threshold = tl.Float(0.0).tag(config=True, sync=True)
#    current_band = tl.Int(10).tag(config=False, sync=True)

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1
    colorstretch = 2.0

    def __init__( self, **kwargs ):   # class_labels: [ [label, RGBA] ... ]
        super(MapManager, self).__init__()
        self._debug = False
        self._pcm_updated = False
        self.norm = None
        self.base: TileServiceBasemap = None
        self._currentFrame = 0
        self.block: Block = None
        self.block_index = None
        self.silent_thresholds = False
        self._adding_marker = False
        self.class_template = None
        self.raster_template = None
        self.points_selection: MarkerManager = None
        self.cluster_selection: ClusterSelector = None
        self.region_selection: PolygonInteractor = None
        self._band_selector: ipw.IntSlider = None
        self._use_model_data = False
        self._cidpress = -1
        self.cspecs=None
        self._classification_data: xa.DataArray = None
        self._class_confidence: xa.DataArray = None
        self.layers = LayersManager( self.on_layer_change )
        self.band_slider: Slider = None
        self.model_slider: Slider = None
        self._spectral_image: Optional[AxesImage] = None
        self.label_map: Optional[xa.DataArray] = None     # Map of classification labels from ML
        self.labels_image: Optional[AxesImage] = None
        self.confidence_image: Optional[AxesImage] = None
        self.clusters_image: Optional[AxesImage] = None
        self.layers.add( 'basemap', 1.0, True)
        self.layers.add( 'bands', 1.0, True )
        self.layers.add( 'markers', 1.0, True )
        self.layers.add( 'labels', 1.0, False )
        self.layers.add( 'confidence', 1.0, False )
        self.layers.add( 'clusters', 1.0   , False )
        self.observe(self.on_threshold_change, names=["lower_threshold", "upper_threshold"])
        self.menu_actions = OrderedDict( Layers = [ [ "Increase Labels Alpha", 'Ctrl+>', None, partial( self.update_image_alpha, "labels", True ) ],
                                                    [ "Decrease Labels Alpha", 'Ctrl+<', None, partial( self.update_image_alpha, "labels", False ) ],
                                                    [ "Increase Band Alpha",   'Alt+>',  None, partial( self.update_image_alpha, "bands", True ) ],
                                                    [ "Decrease Band Alpha",   'Alt+<',  None, partial( self.update_image_alpha, "bands", False ) ] ] )
        self.active_thresholds = ipw.Select( options=[], description='Thresholds:', disabled=False )
        self.active_thresholds.observe( self.on_active_threshold_selection, names=['value'] )
        atexit.register(self.exit)

    def clearMarkers(self):
        for selector in [ self.points_selection, self.cluster_selection, self.region_selection ]:
            selector.clear()

    @exception_handled
    def  on_threshold_change( self,  *args  ):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        if not self.silent_thresholds:
            initialized = self.block.set_thresholds( self._use_model_data, self.currentFrame, (self.lower_threshold, self.upper_threshold) )
            if initialized: self.update_threshold_list()
            pcm().reset()
            self.update_spectral_image()

    def update_thresholds( self ):
        self.silent_thresholds = True
        trec = self.block.get_trec( self._use_model_data, self.currentFrame )
        if trec is not None:
            self.upper_threshold = trec.thresholds[1]
            self.lower_threshold = trec.thresholds[0]
            self.silent_thresholds = False
            lgm().log(f"update_thresholds(frame={self.currentFrame}): [{self.lower_threshold},{self.upper_threshold}]")

    def use_model_data(self, use: bool):
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.learn.manager import ClassificationManager, cm
        if use != self._use_model_data:
            self._use_model_data = use
            if self.base is not None:
                self.update_slider_visibility()
                self.update_spectral_image()
                gpm().use_model_data(use)
                cm().rebuild()

    # @property
    # def band_selector(self):
    #     if self._band_selector is None:
    #         self._band_selector = ipw.IntSlider( self.init_band, 0, self.nFrames(), 1 )
    #         ipw.jslink( (self._band_selector, 'value'), (self.current_band, 'value') )
    #     return self._band_selector
    #
    # def on_current_band_change(self, *args):
    #     lgm().log( f' ** on_current_band_change[{self.current_band}]: args={args}' )
    #     self.update_spectral_image()

    def getPointData(self, **kwargs ) -> xa.DataArray:
        from spectraclass.data.base import DataManager, dm
        current_frame = kwargs.get('current_frame',False)
        if self._use_model_data:
            pdata = dm().getModelData()
        else:
            pdata, coords = self.block.getPointData()
        lgm().log( f" MapManage.getPointData: shape = {pdata.shape}, dims = {pdata.dims}, frame = {self.currentFrame}")
        return pdata[:,self.currentFrame] if current_frame else pdata

    def get_point_coords( self, pid: int ) -> Tuple[float,float]:
        coords = self.block.gid2coords(pid)
        return coords['x'], coords['y']

    @property
    def spectral_image(self) -> Optional[AxesImage]:
        return self._spectral_image

    def get_selection_panel(self):
        self.gui()
        return ipw.Box([self.selection_label, self.selection])

    def get_threshold_panel(self):
        controls, ivals = [], [1.0,0.0]
        for iC, name in enumerate(['upper','lower']):
            slider = ipw.FloatSlider( ivals[iC], description=name, min=0.0, max=1.0, step=0.025 )
            tl.link( (slider, "value"), (self, f'{name}_threshold') )
            controls.append( slider )
        clear_button: ipw.Button = ipw.Button(description="Clear", layout=ipw.Layout(flex='1 1 auto'), border='1px solid dimgrey')
        clear_button.on_click( self.clear_threshold )
        return ipw.HBox( [ipw.VBox( controls ), ipw.VBox( [ self.active_thresholds, clear_button ] )] )

    def update_threshold_list(self):
        options, value = self.block.get_mask_list( self.currentFrame )
        self.active_thresholds.options = options
        if value is not None:
            self.active_thresholds.value = value

    def on_active_threshold_selection( self, *args ):
        active_threshold = self.active_thresholds.value
        if active_threshold is not None:
            [ ttype, sframe ] = active_threshold.split(":")
            self.use_model_data( ttype == "model" )
            self.slider.set_val( int( sframe ) )
            self.update_thresholds()
            self.active_thresholds.value = None

    def clear_threshold(self, *args ):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        trec = self.block.threshold_record( self._use_model_data, self.currentFrame )
        trec.clear()
        self.lower_threshold = 0.0
        self.upper_threshold = 1.0
        self.update_spectral_image()
        pcm().reset()

    def labels_dset(self):
        return xa.Dataset( self.label_map )

    def initLabels(self):
        template = self.block.data[0].squeeze( drop=True )
        self.class_template =  xa.full_like( template, 0, dtype=np.dtype(np.int32) )
        self.raster_template = xa.full_like( template, 0.0, dtype=np.dtype(np.float) )
        self.label_map: xa.DataArray = self.class_template
#        self.label_map.attrs['_FillValue'] = nodata_value
        self.label_map.name = f"{self.block.data.name}_labels"
        self.label_map.attrs[ 'long_name' ] =  "labels"
        self.cspecs = lm().get_labels_colormap()
        self.labels_image = self.class_template.plot.imshow( ax=self.base.gax, alpha=self.layers('labels').visibility, zorder=3.0,
                                                        cmap=self.cspecs['cmap'], add_colorbar=False, norm=self.cspecs['norm'] )
        self.confidence_image = self.raster_template.plot.imshow( ax=self.base.gax, alpha=0.0, zorder=4.0, cmap='jet', add_colorbar=False )
        self.init_cluster_image()

    def clearLabels( self):
        if self.block is not None:
             self.initLabels()
             self.points_selection.plot()
             if self.labels_image is not None:
                self.labels_image.set_alpha(0.0)
                self.confidence_image.set_alpha(0.0)

    def init_cluster_image(self):
         self.clusters_image = self.class_template.plot.imshow( ax=self.base.gax, alpha=self.layers('clusters').visibility, add_colorbar=False, zorder=4.0 )

    @property
    def toolbarMode(self) -> str:
        return self.toolbar.mode

    @property
    def toolbar(self) -> NavigationToolbar2:
        return self.figure.canvas.toolbar

    @exception_handled
    def create_selection_panel(self):
        self.selection_label = ipw.Label( value='Selection Operation:' )
        self.select_modes = [ 'explore', 'select point', 'select region', 'select cluster' ]
        self.selection = ipw.RadioButtons(  options=self.select_modes, disabled=False, layout={'width': 'max-content'} )
        self.selection.observe( self.set_selection_mode, "value" )
        self.points_selection.set_enabled( False )
        self.region_selection.set_enabled( False )
        self.cluster_selection.set_enabled(False)

    @exception_handled
    def set_selection_mode( self, event: Dict ):
        smode = event['new']
        self.set_navigation_enabled(       smode == self.select_modes[0] )
        self.points_selection.set_enabled( smode == self.select_modes[1] )
        self.region_selection.set_enabled( smode == self.select_modes[2] )
        self.cluster_selection.set_enabled(smode == self.select_modes[3] )

    def set_navigation_enabled(self, enabled: bool ):
        from matplotlib.backend_bases import NavigationToolbar2, _Mode
        tbar: NavigationToolbar2 = self.toolbar
        canvas = self.figure.canvas
        for cid in [tbar._id_press, tbar._id_release, tbar._id_drag, self._cidpress ]: canvas.mpl_disconnect(cid)
        if enabled:
            tbar._id_press   = canvas.mpl_connect( 'button_press_event', tbar._zoom_pan_handler )
            tbar._id_release = canvas.mpl_connect( 'button_release_event', tbar._zoom_pan_handler )
            tbar._id_drag    = canvas.mpl_connect( 'motion_notify_event', tbar.point_selection1 )
            self._cidpress   = canvas.mpl_connect( 'button_press_event', self.on_button_press )

    @exception_handled
    def on_button_press(self, event: MouseEvent ):
        from spectraclass.data.spatial.tile.manager import TileManager
        if event.inaxes == self.base.gax:
            c: Dict = self.block.coords2indices( event.ydata, event.xdata )
    #        lgm().log( f" on_button_press: xydata = {(event.xdata,event.ydata)}, c = {(c['ix'],c['iy'])}, transform = {self.block.transform}")
            by, bx = TileManager.reproject_to_latlon(self.block.xcoord[c['ix']], self.block.ycoord[c['iy']] )
            lat,lon = TileManager.reproject_to_latlon( event.xdata, event.ydata )
            msg = f"[{event.y},{event.x}]: ({lat:.4f},{lon:.4f}), block[{c['iy']},{c['ix']}]: ({by:.4f},{bx:.4f})"
            ufm().show( msg, color="blue")
            lgm().log(f" on_button_press: {msg}" )

    @property
    def selectionMode(self) -> str:
        return self.selection.value

    def set_region_class( self, cid: int ):
        if self.region_selection is not None:
            self.region_selection.set_class( cid )

    def create_sliders(self):
        self.band_slider = Slider( ax=self.slider_axes(False), label="band", valmin=0, valmax=self.nFrames(model=False)-1, valstep=1, valfmt="%0.0f" )
        self.band_slider_cid = self.band_slider.on_changed(self._update)
        self.model_slider = Slider( self.slider_axes(True), label="band", valmin=0, valmax=self.nFrames(model=True)-1, valstep=1, valfmt="%0.0f"  )
        self.model_slider_cid = self.model_slider.on_changed(self._update)

    def one_hot_to_index(self, class_data: xa.DataArray, axis=0) -> xa.DataArray:
        return class_data.argmax( axis=axis, skipna=True, keep_attrs=True ).squeeze()

    @property
    def classification_data(self) -> Optional[np.ndarray]:
        return self._classification_data.values if self._classification_data is not None else None

    @property
    def class_confidence(self) -> Optional[np.ndarray]:
        return self._class_confidence.values if self._class_confidence is not None else None

    @exception_handled
    def plot_labels_image(self, classification: xa.DataArray = None, confidence: xa.DataArray = None ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from matplotlib.colors import Normalize
        if classification is None:
            if self._classification_data is not None:
                self._classification_data = xa.zeros_like( self._classification_data )
            if self._class_confidence is not None:
                self._class_confidence = xa.zeros_like( self._class_confidence )
        else:
            lgm().log( f"  plot labels image, shape = {classification.shape}, dims = {classification.dims}")
            self._classification_data = classification.fillna(0.0).squeeze()
            self._class_confidence = confidence
            if self._classification_data.ndim == 3:
                self._classification_data = self.one_hot_to_index( self._classification_data )

        if self._classification_data is not None:
            vrange = [ self._classification_data.values.min(), self._classification_data.values.max() ]
            extent = tm().getBlock().extent
            lgm().log(f"  update labels image, shape={self._classification_data.shape}, vrange={vrange}  ")
            self.labels_image.set_data( self._classification_data.values )
            self.labels_image.set_extent( extent )
            self.labels_image.changed()

            if self._class_confidence is not None:
                cdata = self._class_confidence.values.squeeze()
                nanmask = np.isnan(cdata)
                cdata[ nanmask ] = 0.0
                confdata = cdata # -np.log( 1.0-cdata )
                crange = [ confdata.min(), confdata.max() ]
                h,e = np.histogram( confdata, range=crange, bins=20 )
                lgm().log(f"---> plot confidence image, shape = {confdata.shape}, range = {crange}, nnan = {np.count_nonzero(nanmask)}"
                          f"\n  * histogram = {h.tolist()}"
                          f"\n  * edges = {e.tolist()}")
                self.confidence_image.set_data( confdata )
                self.confidence_image.set_extent( extent )
                self.confidence_image.set_norm( Normalize( *crange ) )
                self.confidence_image.set_alpha( self.layers.alpha("confidence") )
                self.confidence_image.changed()
            self.update_canvas()

    @exception_handled
    def plot_cluster_image(self, clusters: xa.DataArray = None ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if clusters is not None:
            lgm().log( f"Create cluster image, shape={clusters.shape}")
            block = tm().getBlock()
            ncolors = clusters.shape[0]
            color_bounds = np.linspace(-0.5, ncolors - 0.5, ncolors + 1)
            self.clusters_image.set_data( clusters.to_numpy() )
            self.clusters_image.cmap = clusters.attrs['cmap']
            self.clusters_image.norm = mpl.colors.BoundaryNorm( color_bounds, ncolors, clip=True )
            self.clusters_image.set_extent( block.extent )
            self.clusters_image.changed()
            self.update_canvas()

    def layer_managers( self, name: str ) -> List:
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        if   name == "basemap":    mgrs = [ self.base ]
        elif name == "labels":     mgrs = [ self.labels_image ]
        elif name == "confidence": mgrs = [ self.confidence_image ]
        elif name == "clusters":   mgrs = [ self.clusters_image ]
        elif name == "bands":      mgrs = [ self.spectral_image ]
        elif name == "markers":    mgrs = [ self.points_selection, self.region_selection, self.cluster_selection, pcm() ]
        else: raise Exception( f"Unknown Layer: {name}")
        return mgrs

    def initialized(self) -> bool:
        return self.points_selection is not None

    def highlight_points(self, pids: List[int], cids: List[int] ):
        self.points_selection.highlight_points( pids, cids )

    def clear_highlights(self ):
        self.points_selection.clear_highlights()

    def on_layer_change( self, layer: Layer ):
        for mgr in self.layer_managers( layer.name ):
   #         lgm().log( f" **** layer_change[{layer.name}]: {id(mgr)} -> alpha_change[{layer.visibility}]")
            mgr.set_alpha( layer.visibility )
        self.update_canvas()

    @property
    def slider(self) -> Slider:
        return self.model_slider if self._use_model_data else self.band_slider

    @property
    def currentFrame(self):
        return self._currentFrame

    @currentFrame.setter
    def currentFrame(self, value: int ):
        lgm().log( f"MM: Set current Frame: {value}")
        self._currentFrame = value
        self.update_thresholds()
        self.update_spectral_image()

    @exception_handled
    def _update( self, val: float ):
        self.currentFrame = int( val )

    def update_image_alpha( self, layer: str, increase: bool, *args, **kwargs ):
        self.layers(layer).increment( increase )

    def get_color_bounds( self, raster: xa.DataArray ):
        ave = raster.mean(skipna=True).values
        std = raster.std(skipna=True).values
        if std == 0.0:
            msg =  "This block does not appear to contain any data.  Suggest trying a different tile/block."
            ufm().show( msg, "red" ); lgm().log( "\n" +  msg + "\n"  )
        return dict( vmin= ave - std * self.colorstretch, vmax= ave + std * self.colorstretch  )

    @exception_handled
    def update_spectral_image(self):
        if self.base is not None:
            fdata: xa.DataArray = self.frame_data
            if fdata is not None:
                lgm().log(f"set_color_bounds: full data range = {[np.nanmin(fdata.values),np.nanmax(fdata.values)]}")
                drange = self.get_color_bounds(fdata)
                alpha = self.layers('bands').visibility
                self.norm = Normalize(**drange)
                if self._spectral_image is None:
                    self.base.set_bounds(self.block.xlim, self.block.ylim)
                    self._spectral_image: AxesImage = fdata.plot.imshow(ax=self.base.gax, alpha=alpha, cmap='jet', norm=self.norm, add_colorbar=False, zorder=2.0 )
                else:
                    self._spectral_image.set_norm( self.norm )
                    self._spectral_image.set_data(fdata.values)
                    self._spectral_image.set_alpha(alpha)
#                with self.base.hold_limits():
                self._spectral_image.set_extent(self.block.extent)
                self.update_canvas()
                self._pcm_updated = False

                lgm().log(f"UPDATE spectral_image({id(self._spectral_image)}): data shape = {fdata.shape}, drange={drange}, "
                          f"xlim={fs(self.block.xlim)}, ylim={fs(self.block.ylim)}, model_data={self._use_model_data} " )

            else: lgm().log(f"UPDATE spectral_image: fdata is None")
        else: lgm().log(f"UPDATE spectral_image: base is None")

    def update_pcm(self):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        fdata: xa.DataArray = self.frame_data
        if not self._pcm_updated and (fdata is not None):
            pcm().update_plot(cdata=fdata, norm=self.norm)
            self._pcm_updated = True

    def reset_plot(self):
        from spectraclass.data.base import DataManager, dm
#        self._spectral_image.remove()
#        self._spectral_image = None
        plot_name = os.path.basename( dm().dsid() )
        if self.plot_axes is not None:
            self.plot_axes.title.set_text(f"{plot_name}: Band {self.currentFrame + 1}")
            self.plot_axes.title.set_fontsize(8)
        self.setBlock()

    @exception_handled
    def update_plots(self, new_image=False):
        from spectraclass.data.base import DataManager, dm
        if new_image:  dm().modal.update_extent()
        self.reset_plot()
        self.update_thresholds()
        self.update_spectral_image()

    def update_canvas(self):
        self.figure.canvas.draw_idle()

    def nFrames(self, **kwargs ) -> int:
        from spectraclass.data.base import DataManager, dm
        use_model = kwargs.get( 'model', self._use_model_data )
        return dm().getModelData().shape[1] if use_model else self.data.shape[0]

    @property
    def frame_data(self) -> Optional[xa.DataArray]:
        if self.currentFrame >= self.nFrames(): return None
        fdata = self.data[self.currentFrame]
        vrange = [ np.nanmin(fdata.values), np.nanmax(fdata.values) ]
        lgm().log( f" ******* frame_data[{self.currentFrame}], shape: {fdata.shape}, {svalid(fdata.values)}, vrange={vrange}, attrs={fdata.attrs.keys()}")
        tmask: np.ndarray = self.block.get_threshold_mask( raster=True )
        if tmask is None:
            lgm().log(f" ---> NO threshold mask")
        else:
            mdata = fdata.values.flatten()
            mdata[(~tmask).flatten()] = np.nan
            fdata = fdata.copy( data=mdata.reshape(fdata.shape) )
            lgm().log(f" ---> threshold mask, {svalid(fdata.values)}")
        return fdata

    def threshold_mask( self, raster=True ) -> np.ndarray:
        return None if (self.block is None) else self.block.get_threshold_mask(raster)

    @property
    def figure(self) -> Figure:
        return self.base.figure

    @property
    def plot_axes(self) -> Optional[Axes]:
        return None if (self.base is None) else self.base.gax

    def slider_axes(self, use_model = False ) -> Axes:
        return self.base.msax if use_model else self.base.bsax

    def update_slider_visibility(self):
        lgm().log( f" UPDATE MAP SLIDER: model = {self._use_model_data}")
        self.base.update_slider_visibility( self._use_model_data )

    def invert_yaxis(self):
        self.plot_axes.invert_yaxis()

    def get_xy_coords(self,  ) -> Tuple[ np.ndarray, np.ndarray ]:
        return self.get_coord(self.x_axis ), self.get_coord( self.y_axis )

    def get_anim_coord(self ) -> np.ndarray:
        return self.get_coord( 0 )

    def get_coord(self,   iCoord: int ) -> np.ndarray:
        return self.data.coords[  self.data.dims[iCoord] ].values

    def image_update(self):
        self.block = None
        self.setBlock()

    @property
    def data(self) -> Optional[xa.DataArray]:
        return self.getRasterData( self._use_model_data )

    def getRasterData(self, use_model: bool ) -> Optional[xa.DataArray]:
        from spectraclass.data.base import dm
        if self.block is None: self.setBlock()
        return self.block.points2raster( dm().getModelData() ) if use_model else self.block.data

    def getModelData(self, raster: bool ):
        from spectraclass.data.base import dm
        if self.block is None: self.setBlock()
        point_data = dm().getModelData()
        return self.block.points2raster(  point_data ) if raster else point_data

    @exception_handled
    def setBlock( self, block_index: Tuple[int,int] = None, **kwargs ):
        from spectraclass.data.spatial.tile.manager import tm
        from spectraclass.learn.cluster.manager import clm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        needs_update = (self.block is None) if (block_index is None) else (tuple(block_index) != tuple(tm().block_index))
        if needs_update:
            if block_index is None: block_index = tm().block_index
            lgm().trace( f"Loading Block[{tm().image_index}:{tm().image_name}]{block_index}, tm.index={tm().block_index}, current.index={self.block_index}")
            tm().setBlock( block_index )
            self.block: Block = tm().getBlock()
            if (self.block is not None):
                ufm().show(f" --> Loading Block{self.block.block_coords}" )
                t0 = time.time()
                self.block_index = block_index
                dm().clear_project_cache()
                pcm().reset()
                update = kwargs.get( 'update', False )
                lgm().log(f" -------------------- Loading block: {self.block.block_coords}  -------------------- " )
                if self.base is not None:
                    self.base.set_bounds(self.block.xlim, self.block.ylim)
                self.band_axis = kwargs.pop('band', 0)
                self.z_axis_name = self.data.dims[self.band_axis]
                self.x_axis = kwargs.pop('x', 2)
                self.x_axis_name = self.data.dims[self.x_axis]
                self.y_axis = kwargs.pop('y', 1)
                self.y_axis_name = self.data.dims[self.y_axis]
                gpm().refresh()
                clm().clear()
                t1 = time.time()
                if update:  self.update_plots()
                t2 = time.time()
                ufm().show(f" ** Block Loaded: {t1-t0:.2f} {t2-t1:.2f} ")

    def gui(self,**kwargs):
        if self.base is None:
            self.setBlock()
            [x0, x1, y0, y1] = self.block.extent
            self.base = TileServiceBasemap()
            standalone = self.base.setup_plot("Label Construction", (x0, x1), (y0, y1), index=100, **kwargs)
            self.init_map()
            self.region_selection = PolygonInteractor(self.base.gax)
            self.points_selection = MarkerManager(self.base.gax)
            self.cluster_selection = ClusterSelector(self.base.gax)
            self.init_hover()
            if not standalone:
                self.create_selection_panel()
        return self.base.gax.figure.canvas

    def mark_point(self, pid: int, **kwargs ) -> Optional[Tuple[float,float]]:
        point = self.points_selection.mark_point( pid, **kwargs )
        return point

    def init_hover(self):
        def format_coord(x, y):
            return "x: {}, y: {}".format(x, y)
        self.base.gax.format_coord = format_coord

    def plot_markers_image(self, **kwargs ):
        self.points_selection.plot( **kwargs )

    def init_map(self):
        from spectraclass.data.base import DataManager, dm
        self.update_spectral_image()
        self.create_sliders()
        self.initLabels()
        self._cidpress = self.figure.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.use_model_data( dm().use_model_data )
     #   self._cidrelease = self._spectral_image.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease )
     #   self.plot_axes.callbacks.connect('ylim_changed', self.on_lims_change)

    def __del__(self):
        self.exit()

    def exit(self):
        pass

if __name__ == '__main__':
    from spectraclass.data.base import DataManager, dm
    dmgr: DataManager = DataManager.initialize("demo2", 'desis')
    dmgr.loadCurrentProject("main")
    classes = [('Class-1', "cyan"), ('Class-2', "green"), ('Class-3', "magenta"), ('Class-4', "blue")]
    lm().setLabels(classes)

    mm = MapManager()
    panel = mm.gui()
    plt.show( )


