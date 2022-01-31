import xarray as xa
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import logging, os
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
from spectraclass.widgets.slider import PageSlider
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.data.spatial.tile.tile import Block, Tile

def mm(**kwargs) -> "MapManager":
    return MapManager.instance(**kwargs)

def fs(flist):
    return [f"{fv:.1f}" for fv in flist]

class MapManager(SCSingletonConfigurable):
    init_band = tl.Int(10).tag(config=True, sync=True)

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1
    colorstretch = 2.0

    def __init__( self, **kwargs ):   # class_labels: [ [label, RGBA] ... ]
        super(MapManager, self).__init__()
        self._debug = False
        self.base: TileServiceBasemap = None
        self.currentFrame = 0
        self.block: Block = None
        self._adding_marker = False
        self.points_selection: MarkerManager = None
        self._use_model_data = False
        self._cidpress = -1
        self.cspecs=None
        self._classification_data: xa.DataArray = None
        self.layers = LayersManager( self.on_layer_change )
        self.band_slider: PageSlider = None
        self.model_slider: PageSlider = None
        self._spectral_image: Optional[AxesImage] = None
        self.label_map: Optional[xa.DataArray] = None     # Map of classification labels from ML
        self.region_selection: PolygonInteractor = None
        self.labels_image: Optional[AxesImage] = None
        self.layers.add( 'basemap', 1.0, True)
        self.layers.add( 'bands', 1.0, True )
        self.layers.add( 'markers', 0.5, True )
        self.layers.add( 'labels', 0.5, False )
        self.menu_actions = OrderedDict( Layers = [ [ "Increase Labels Alpha", 'Ctrl+>', None, partial( self.update_image_alpha, "labels", True ) ],
                                                    [ "Decrease Labels Alpha", 'Ctrl+<', None, partial( self.update_image_alpha, "labels", False ) ],
                                                    [ "Increase Band Alpha",   'Alt+>',  None, partial( self.update_image_alpha, "bands", True ) ],
                                                    [ "Decrease Band Alpha",   'Alt+<',  None, partial( self.update_image_alpha, "bands", False ) ] ] )
        atexit.register(self.exit)

    def use_model_data(self, use: bool ):
        self._use_model_data = use
        if self.base is not None:
            self.update_slider_visibility()
            self.update_spectral_image()

    def getPointData(self, **kwargs ) -> xa.DataArray:
        from spectraclass.data.base import DataManager, dm
        current_frame = kwargs.get('current_frame',False)
        if self._use_model_data:
            pdata = dm().getModelData()
        else:
            pdata, coords = self.block.getPointData()
        lgm().log( f" MapManage.getPointData: shape = {pdata.shape}, dims = {pdata.dims}")
        return pdata[:,self.currentFrame] if current_frame else pdata

    def get_point_coords( self, pid: int ) -> Tuple[float,float]:
        coords = self.block.pid2coords(pid)
        return coords['x'], coords['y']

    @property
    def spectral_image(self) -> Optional[AxesImage]:
        return self._spectral_image

    def get_selection_panel(self):
        self.gui()
        return ipw.Box([self.selection_label, self.selection])

    def labels_dset(self):
        return xa.Dataset( self.label_map )

    def initLabels(self):
        nodata_value = -2
        template = self.block.data[0].squeeze( drop=True )
        self.label_map: xa.DataArray = xa.full_like( template, 0, dtype=np.dtype(np.int32) ) # .where( template.notnull(), nodata_value )
#        self.label_map.attrs['_FillValue'] = nodata_value
        self.label_map.name = f"{self.block.data.name}_labels"
        self.label_map.attrs[ 'long_name' ] =  "labels"
        self.cspecs = lm().get_labels_colormap()
        lgm().log( f"\nInit label map, value range = [{self.label_map.values.min()} {self.label_map.values.max()}]\n")
        self.labels_image = self.label_map.plot.imshow( ax=self.base.gax, alpha=self.layers('labels').visibility,
                                                        cmap=self.cspecs['cmap'], add_colorbar=False, norm=self.cspecs['norm'] )

    def clearLabels( self):
        if self.block is not None:
             self.initLabels()
             self.points_selection.plot()
             if self.labels_image is not None:
                self.labels_image.set_alpha(0.0)

    @property
    def toolbarMode(self) -> str:
        return self.toolbar.mode

    @property
    def toolbar(self) -> NavigationToolbar2:
        return self.figure.canvas.toolbar

    @exception_handled
    def create_selection_panel(self):
        self.selection_label = ipw.Label(value='Selection Operation:')
        self.select_modes = [ 'explore', 'select point', 'select region' ]
        self.selection = ipw.RadioButtons(  options=self.select_modes, disabled=False, layout={'width': 'max-content'} )
        self.selection.observe( self.set_selection_mode, "value" )
        self.points_selection.set_enabled( False )
        self.region_selection.set_enabled( False )

    @exception_handled
    def set_selection_mode( self, event: Dict ):
        smode = event['new']
        self.set_navigation_enabled(       smode == self.select_modes[0] )
        self.points_selection.set_enabled( smode == self.select_modes[1] )
        self.region_selection.set_enabled( smode == self.select_modes[2] )

    def set_navigation_enabled(self, enabled: bool ):
        from matplotlib.backend_bases import NavigationToolbar2, _Mode
        tbar: NavigationToolbar2 = self.toolbar
        canvas = self.figure.canvas
        for cid in [tbar._id_press, tbar._id_release, tbar._id_drag, self._cidpress ]: canvas.mpl_disconnect(cid)
        if enabled:
            tbar._id_press   = canvas.mpl_connect( 'button_press_event', tbar._zoom_pan_handler )
            tbar._id_release = canvas.mpl_connect( 'button_release_event', tbar._zoom_pan_handler )
            tbar._id_drag    = canvas.mpl_connect( 'motion_notify_event', tbar.mouse_move )
            self._cidpress   = canvas.mpl_connect( 'button_press_event', self.on_button_press )

    @exception_handled
    def on_button_press(self, event: MouseEvent ):
        from spectraclass.data.spatial.tile.manager import TileManager
        if event.inaxes == self.base.gax:
            c: Dict = self.block.coords2indices( event.ydata, event.xdata )
    #        lgm().log( f" on_button_press: xydata = {(event.xdata,event.ydata)}, c = {(c['ix'],c['iy'])}, transform = {self.block.transform}")
            by, bx = TileManager.reproject_to_latlon(self.block.xcoord[c['ix']], self.block.ycoord[c['iy']] )
            lat,lon = TileManager.reproject_to_latlon( event.xdata, event.ydata )
            ufm().show( f"[{event.y},{event.x}]: ({lat:.4f},{lon:.4f}), block[{c['iy']},{c['ix']}]: ({by:.4f},{bx:.4f})", color="blue")

    @property
    def selectionMode(self) -> str:
        return self.selection.value

    def set_region_class(self, cid: int ):
        self.region_selection.set_class( cid )

    def create_sliders(self):
        self.band_slider = PageSlider( self.slider_axes(False), self.nFrames(model=False) )
        self.band_slider_cid = self.band_slider.on_changed(self._update)
        self.model_slider = PageSlider( self.slider_axes(True), self.nFrames(model=True) )
        self.model_slider_cid = self.model_slider.on_changed(self._update)

    def one_hot_to_index(self, class_data: xa.DataArray) -> xa.DataArray:
        return class_data.argmax( axis=0, skipna=True, keep_attrs=True ).squeeze()

    @exception_handled
    def plot_labels_image(self, classification: xa.DataArray = None ):

        if classification is None:
            if self._classification_data is not None:
                self._classification_data = xa.zeros_like( self._classification_data )
        else:
            self._classification_data = classification.fillna(0.0).squeeze()
            if self._classification_data.ndim == 3:
                self._classification_data = self.one_hot_to_index( self._classification_data )

        if self._classification_data is not None:
            vrange = [ self._classification_data.values.min(), self._classification_data.values.max() ]
            lgm().log( f"\n plot labels image, shape = {self._classification_data.shape}, vrange = {vrange}\n" )
            try: self.labels_image.remove()
            except Exception: pass
            self.labels_image = self._classification_data.plot.imshow(ax=self.base.gax, alpha=0.5, cmap=self.cspecs['cmap'],
                                                           add_colorbar=False, norm=self.cspecs['norm'])
            self.layers.set_visibility( "labels", 0.5, True, notify=False )
            self.update_canvas()

    def layer_managers( self, name: str ) -> List:
        if name == "basemap":  mgrs = [self.base]
        elif name   == "labels":  mgrs = [ self.labels_image ]
        elif name == "bands":   mgrs = [ self.spectral_image ]
        elif name == "markers": mgrs = [ self.points_selection, self.region_selection ]
        else: raise Exception( f"Unknown Layer: {name}")
        return mgrs

    def highlight_points(self, pids: List[int], cids: List[int] ):
        self.points_selection.highlight_points( pids, cids )

    def clear_highlights(self ):
        self.points_selection.clear_highlights()

    def on_layer_change( self, layer: Layer ):
        for mgr in self.layer_managers( layer.name ):
 #           lgm().log( f" **** layer_change[{layer.name}]: {id(mgr)} -> alpha_change[{layer.visibility}]")
            mgr.set_alpha( layer.visibility )
        self.update_canvas()

    @property
    def slider(self) -> PageSlider:
        return self.model_slider if self._use_model_data else self.band_slider

    @exception_handled
    def _update( self, val ):
        self.currentFrame = int( self.slider.val )
        self.slider.refesh()
        self.update_plots()

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
            self.base.set_bounds(self.block.xlim, self.block.ylim)
            fdata: xa.DataArray = self.frame_data
            if fdata is not None:
                drange = self.get_color_bounds(fdata)
                try: self._spectral_image.remove()
                except Exception: pass
                self._spectral_image: AxesImage = fdata.plot.imshow(ax=self.base.gax, alpha=self.layers('bands').visibility, cmap='jet', norm=Normalize(**drange), add_colorbar=False)
                lgm().log(f"\n CREATE spectral_image({id(self._spectral_image)}): data shape = {fdata.shape}, drange={drange}, xlim={fs(self.block.xlim)}, ylim={fs(self.block.ylim)}" )
                self.update_canvas()

    @exception_handled
    def update_plots(self, **kwargs ):
        from spectraclass.gui.points3js import PointCloudManager, pcm
        from spectraclass.data.spatial.manager import SpatialDataManager
        from spectraclass.data.base import DataManager, dm
        new_image = kwargs.get( 'new_image', None )
        if new_image is not None:
            self.block = None
            self._spectral_image.remove()
            dm().modal.update_extent()
            lgm().log(f"\n <------> Loading new image: {os.path.basename(new_image)} <------> \n")
        if self._spectral_image is not None:
            fdata: xa.DataArray = self.frame_data
            lgm().log(f"update_plots: block data shape = {self.data.shape}" )
            if fdata is not None:
                drange = self.get_color_bounds(fdata)
                alpha = self.layers('bands').visibility
                norm = Normalize(**drange)
                self._spectral_image.set_data(fdata.values)
                self._spectral_image.set_norm(norm)
                self._spectral_image.set_alpha(alpha)
                plot_name = os.path.basename(dm().dsid())
                self.plot_axes.title.set_text(f"{plot_name}: Band {self.currentFrame+1}" )
                self.plot_axes.title.set_fontsize( 8 )
                self.update_canvas()
                lgm().log(f" --> AXIS: xlim={fs(self.plot_axes.get_xlim())}, ylim={fs(self.plot_axes.get_ylim())}")
                lgm().log(f" --> DATA: extent={fs(SpatialDataManager.extent(fdata))}")
                pcm().update_plot(cdata=fdata, norm=norm)

    def update_canvas(self):
        self.figure.canvas.draw_idle()

    def nFrames(self, **kwargs ) -> int:
        from spectraclass.data.base import DataManager, dm
        use_model = kwargs.get( 'model', self._use_model_data )
        return dm().getModelData().shape[1] if use_model else self.data.shape[0]

    @property
    def frame_data(self) -> Optional[xa.DataArray]:
        if self.currentFrame >= self.nFrames(): return None
        # lgm().log( f" color_pointcloud: currentFrame = {self.currentFrame}, frame data shape = {frame_data.shape}")
        # app().color_pointcloud( frame_data.values.flatten(), **kwargs )
        return self.data[self.currentFrame]

    @property
    def figure(self) -> Figure:
        return self.base.figure

    @property
    def plot_axes(self) -> Axes:
        return self.base.gax

    def slider_axes(self, use_model = False ) -> Axes:
        return self.base.msax if use_model else self.base.bsax

    def update_slider_visibility(self):
        self.base.msax.set_visible( self._use_model_data )
        self.base.bsax.set_visible( not self._use_model_data )

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

    @property
    def data(self) -> Optional[xa.DataArray]:
        from spectraclass.data.base import dm
        if self.block is None: self.setBlock()
        return self.block.points2raster( dm().getModelData() ) if self._use_model_data else self.block.data

    @exception_handled
    def setBlock( self, block_index: Tuple[int,int] = None, **kwargs ):
        from spectraclass.data.spatial.tile.manager import tm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        self.block: Block = tm().getBlock( index=block_index )
        if self.block is not None:
            update = kwargs.get( 'update', False )
            lgm().log(f"\n -------------------- Loading block: {self.block.block_coords}  -------------------- " )
            self.update_spectral_image()
            if self.points_selection is not None:
                self.points_selection.set_block(self.block)
            self.band_axis = kwargs.pop('band', 0)
            self.z_axis_name = self.data.dims[self.band_axis]
            self.x_axis = kwargs.pop('x', 2)
            self.x_axis_name = self.data.dims[self.x_axis]
            self.y_axis = kwargs.pop('y', 1)
            self.y_axis_name = self.data.dims[self.y_axis]
            gpm().refresh()
            if update: self.update_plots()

    def gui(self,**kwargs):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if self.base is None:
            self.setBlock()
            self.base = TileServiceBasemap()
            [x0, x1, y0, y1] = self.block.extent
            standalone = self.base.setup_plot( "Label Construction", (x0,x1), (y0,y1), index=100, **kwargs )
            self.init_map()
            self.region_selection = PolygonInteractor( self.base.gax )
            self.points_selection = MarkerManager( self.base.gax, self.block )
            self.init_hover()
            if not standalone:
                self.create_selection_panel()
        return self.base.gax.figure.canvas

    def mark_point(self, pid: int, **kwargs ) -> Optional[Tuple[float,float]]:
        return self.points_selection.mark_point( pid, **kwargs )

    def init_hover(self):
        def format_coord(x, y):
            return "x: {}, y: {}".format(x, y)
        self.base.gax.format_coord = format_coord

    def plot_markers_image(self, **kwargs ):
        self.points_selection.plot( **kwargs )

    def init_map(self):
        self.update_spectral_image()
        self.create_sliders()
        self.initLabels()
        self._cidpress = self.figure.canvas.mpl_connect('button_press_event', self.on_button_press)
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


