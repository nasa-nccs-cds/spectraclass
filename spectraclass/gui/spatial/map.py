import xarray as xa
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import logging, os
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

class MapManager(SCSingletonConfigurable):
    init_band = tl.Int(10).tag(config=True, sync=True)

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1
    colorstretch = 2.0

    def __init__( self, **kwargs ):   # class_labels: [ [label, RGBA] ... ]
        super(MapManager, self).__init__()
        self._debug = False
        self.base = None
        self.currentFrame = 0
        self.block: Block = None
        self._adding_marker = False
        self.points_selection: MarkerManager = None
        self.use_model_data = False
        self._cidpress = -1
        self._cidrelease = -1
        self._classification_data = None
        self.layers = LayersManager( self.on_layer_change )
        self.slider: Optional[PageSlider] = None
        self.image: Optional[AxesImage] = None
        self.label_map: Optional[xa.DataArray] = None     # Map of classification labels from ML
        self.region_selection: PolygonInteractor = None
        self.labels_image: Optional[AxesImage] = None
        self.layers.add( 'bands', 1.0, True )
        self.layers.add( 'markers', 0.5, False )
        self.layers.add( 'labels', 0.5, False )
        self.menu_actions = OrderedDict( Layers = [ [ "Increase Labels Alpha", 'Ctrl+>', None, partial( self.update_image_alpha, "labels", True ) ],
                                                    [ "Decrease Labels Alpha", 'Ctrl+<', None, partial( self.update_image_alpha, "labels", False ) ],
                                                    [ "Increase Band Alpha",   'Alt+>',  None, partial( self.update_image_alpha, "bands", True ) ],
                                                    [ "Decrease Band Alpha",   'Alt+<',  None, partial( self.update_image_alpha, "bands", False ) ] ] )
        atexit.register(self.exit)

    def get_selection_panel(self):
        self.gui()
        return ipw.Box([self.selection_label, self.selection])

    def labels_dset(self):
        return xa.Dataset( self.label_map )

    def initLabels(self):
        nodata_value = -2
        template = self.block.data[0].squeeze( drop=True )
        self.label_map: xa.DataArray = xa.full_like( template, 0, dtype=np.dtype(np.int32) ).where( template.notnull(), nodata_value )
        self.label_map.attrs['_FillValue'] = nodata_value
        self.label_map.name = f"{self.block.data.name}_labels"
        self.label_map.attrs[ 'long_name' ] =  "labels"
        cspecs = lm().get_labels_colormap()
        self.labels_image = self.label_map.plot.imshow( ax=self.base.gax, alpha=self.layers('labels').visibility,
                                                        cmap=cspecs['cmap'], add_colorbar=False, norm=cspecs['norm'] )

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
        for cid in [tbar._id_press, tbar._id_release, tbar._id_drag]: canvas.mpl_disconnect(cid)
        if enabled:
            tbar._id_press   = canvas.mpl_connect( 'button_press_event', tbar._zoom_pan_handler )
            tbar._id_release = canvas.mpl_connect( 'button_release_event', tbar._zoom_pan_handler )
            tbar._id_drag    = canvas.mpl_connect( 'motion_notify_event', tbar.mouse_move )

    @property
    def selectionMode(self) -> str:
        return self.selection.value

    def set_region_class(self, cid: int ):
        self.region_selection.set_class( cid )

    def add_slider(self,  **kwargs ):
        if self.slider is None:
            self.slider = PageSlider( self.slider_axes, self.nFrames )
            self.slider_cid = self.slider.on_changed(self._update)

    @exception_handled
    def plot_labels_image(self):
        self._classification_data = lm().get_label_map( self.block )
        lgm().log( f"\n plot labels image, shape = {self._classification_data.shape}, vrange = {[ self._classification_data.min(), self._classification_data.max() ]}\n" )
        self.labels_image.set_data( self._classification_data )
        self.labels_image.set_alpha( self.layers( 'labels' ).visibility )
        self.update_canvas()

    def layer_image( self, name: str ):
        if name   == "labels":  img = self.labels_image
        elif name == "bands":   img = self.image
        elif name == "markers": img = self.points_selection.points
        else: raise Exception( f"Unknown Layer: {name}")
        return img

    def on_layer_change( self, layer: Layer ):
        image = self.layer_image( layer.name )
        image.set_alpha( layer.visibility )
#        lgm().log(f" image {layer.name} set alpha = {layer.visibility}" )
        self.update_canvas()

    @exception_handled
    def _update( self, val ):
        if self.slider is not None:
            tval = self.slider.val
            self.currentFrame = int( tval )
            lgm().log(f"Slider Update, frame = {self.currentFrame}")
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
    def update_plots(self):
        from spectraclass.data.base import DataManager, dm
        if self.image is not None:
            fdata: xa.DataArray = self.frame_data
            if fdata is not None:
                self.image.set_data(fdata.values)
                self.image.set_alpha(self.layers('bands').visibility)
                drange = self.get_color_bounds( fdata )
                self.image.set_norm( Normalize( **drange ) )
                plot_name = os.path.basename(dm().dsid())
                self.plot_axes.title.set_text(f"{plot_name}: Band {self.currentFrame+1}" )
                self.plot_axes.title.set_fontsize( 8 )
                self.update_canvas()

    def update_canvas(self):
        self.figure.canvas.draw_idle()

    @property
    def frame_data(self) -> Optional[xa.DataArray]:
        if self.currentFrame >= self.data.shape[0]: return None
        # lgm().log( f" color_pointcloud: currentFrame = {self.currentFrame}, frame data shape = {frame_data.shape}")
        # app().color_pointcloud( frame_data.values.flatten(), **kwargs )
        return self.block.data[self.currentFrame].squeeze(drop=True)

    @property
    def figure(self) -> Figure:
        return self.base.figure

    @property
    def plot_axes(self) -> Axes:
        return self.base.gax

    @property
    def slider_axes(self) -> Axes:
        return self.base.sax

    def invert_yaxis(self):
        self.plot_axes.invert_yaxis()

    def get_xy_coords(self,  ) -> Tuple[ np.ndarray, np.ndarray ]:
        return self.get_coord(self.x_axis ), self.get_coord( self.y_axis )

    def get_anim_coord(self ) -> np.ndarray:
        return self.get_coord( 0 )

    def get_coord(self,   iCoord: int ) -> np.ndarray:
        return self.data.coords[  self.data.dims[iCoord] ].values

    @property
    def data(self) -> Optional[xa.DataArray]:
        from spectraclass.data.base import DataManager, dm
        if self.block is None: self.setBlock()
        block_data: xa.DataArray = self.block.data
        if self.use_model_data:
            reduced_data: xa.DataArray = dm().getModelData().transpose()
            dims = [reduced_data.dims[0], block_data.dims[1], block_data.dims[2]]
            coords = [(dims[0], reduced_data[dims[0]]), (dims[1], block_data[dims[1]]), (dims[2], block_data[dims[2]])]
            shape = [c[1].size for c in coords]
            raster_data = reduced_data.data.reshape(shape)
            return xa.DataArray(raster_data, coords, dims, reduced_data.name, reduced_data.attrs)
        else:
            return block_data

    def setBlock( self, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager
        tm = TileManager.instance()
        self.block: Block = tm.getBlock()
        if self.block is not None:
            if self.points_selection is not None:
                self.points_selection.set_block(self.block)
            self.nFrames = self.data.shape[0]
            self.band_axis = kwargs.pop('band', 0)
            self.z_axis_name = self.data.dims[self.band_axis]
            self.x_axis = kwargs.pop('x', 2)
            self.x_axis_name = self.data.dims[self.x_axis]
            self.y_axis = kwargs.pop('y', 1)
            self.y_axis_name = self.data.dims[self.y_axis]

    def gui(self,**kwargs):
        if self.base is None:
            self.setBlock()
            self.base = TileServiceBasemap()
            [x0, x1, y0, y1] = self.block.extent()
            self.base.setup_plot( (x0,x1), (y0,y1), **kwargs )
            self.init_map(**kwargs)
            self.region_selection = PolygonInteractor( self.base.gax )
            self.points_selection = MarkerManager( self.base.gax, self.block )
            self.create_selection_panel()
        return self.base.gax.figure.canvas

    def plot_markers_image(self):
        self.points_selection.plot()

    def init_map(self,**kwargs):
        self.image: AxesImage = self.frame_data.plot.imshow( ax=self.base.gax, alpha=self.layers('bands').visibility, cmap='jet' )
        self.add_slider(**kwargs)
        self.initLabels()
     #   self._cidrelease = self.image.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease )
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


