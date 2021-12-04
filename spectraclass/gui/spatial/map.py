import xarray as xa
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled
import logging, os
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
from spectraclass.widgets.polygons import PolygonInteractor, Polygon
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
from spectraclass.model.base import SCSingletonConfigurable, Marker
from spectraclass.data.spatial.tile.tile import Block, Tile

def mm(**kwargs) -> "MapManager":
    return MapManager.instance(**kwargs)

class MapManager(SCSingletonConfigurable):
    init_band = tl.Int(10).tag(config=True, sync=True)

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1

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
        self.layers = LayersManager( self.on_layer_change )
        self.slider: Optional[PageSlider] = None
        self.image: Optional[AxesImage] = None
        self.label_map: Optional[xa.DataArray] = None     # Map of classification labels from ML
        self.region_selection: PolygonInteractor = None
        self.labels_image: Optional[AxesImage] = None
        self.layers.add('bands', 1.0, True)
        self.layers.add('labels', 0.5, False)
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
        self.label_map: xa.DataArray = xa.full_like( template, -1, dtype=np.dtype(np.int32) ).where( template.notnull(), nodata_value )
        self.label_map.attrs['_FillValue'] = nodata_value
        self.label_map.name = f"{self.block.data.name}_labels"
        self.label_map.attrs[ 'long_name' ] = [ "labels" ]

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
        self.select_modes = ['select point', 'select region']
        self.selection = ipw.RadioButtons(  options=self.select_modes, disabled=False, layout={'width': 'max-content'} )
        self.selection.observe( self.set_selection_mode, "value" )
        self.points_selection.set_enabled( True )

    @exception_handled
    def set_selection_mode( self, event: Dict ):
        smode = event['new']
        self.points_selection.set_enabled( smode == self.select_modes[0] )
        self.region_selection.set_enabled( smode == self.select_modes[1] )

    @property
    def selectionMode(self) -> str:
        return self.selection.value

    def set_region_color(self, color: str ):
        self.region_selection.set_color(color)

    def add_slider(self,  **kwargs ):
        if self.slider is None:
            self.slider = PageSlider( self.slider_axes, self.nFrames )
            self.slider_cid = self.slider.on_changed(self._update)

    @exception_handled
    def plot_overlay_image( self, image_data: np.ndarray = None ):
        if image_data is not None:
            lgm().log( f" plot image overlay, shape = {image_data.shape}, vrange = {[ image_data.min(), image_data.max() ]}, dtype = {image_data.dtype}" )
            self._classification_data = image_data
            self.labels_image.set_data(image_data)
        self.labels_image.set_alpha(self.layers('overlay').visibility)
        self.update_canvas()

    def layer_image( self, name: str ):
        if name   == "labels": img = self.labels_image
        elif name == "bands":  img = self.image
        else: raise Exception( f"Unknown Layer: {name}")
        return img

    def on_layer_change( self, layer: Layer ):
        image = self.layer_image( layer.name )
        image.set_alpha( layer.visibility )
        lgm().log(f" image {layer.name} set alpha = {layer.visibility}" )
        self.update_canvas()

    @exception_handled
    def _update( self, val ):
        if self.slider is not None:
            tval = self.slider.val
            self.currentFrame = int( tval )
            lgm().log(f"Slider Update, frame = {self.currentFrame}")
#            ufm().show( f"Loading frame {self.currentFrame}", "yellow" )
            self.update_plots()
#            ufm().clear()

    def update_image_alpha( self, layer: str, increase: bool, *args, **kwargs ):
        self.layers(layer).increment( increase )

    @exception_handled
    def update_plots(self):
        from spectraclass.data.base import DataManager, dm
        if self.image is not None:
            fdata: xa.DataArray = self.frame_data
            if fdata is not None:
                self.image.set_data(fdata.values)
                self.image.set_alpha(self.layers('bands').visibility)
#                drange = dms().get_color_bounds( frame_data )
#                self.image.set_norm( Normalize( **drange ) )
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
        self.image: AxesImage = self.frame_data.plot.imshow( ax=self.base.gax, alpha=self.layers('bands').visibility )
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


