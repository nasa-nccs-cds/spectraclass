import xarray as xa
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled
import logging, os
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from spectraclass.gui.control import UserFeedbackManager, ufm
from matplotlib.axes import Axes
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.gui.spatial.basemap import TileServiceBasemap
from spectraclass.widgets.polygons import PolygonInteractor, Polygon
import matplotlib.pyplot as plt
import ipywidgets as ipw
from spectraclass.data.base import DataManager
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

    def __init__( self, **kwargs ):   # class_labels: [ [label, RGBA] ... ]
        super(MapManager, self).__init__()
        self._debug = False
        self.currentFrame = 0
        self.block: Block = None
        self.use_model_data = False
        self.slider: Optional[PageSlider] = None
        self.image: Optional[AxesImage] = None
        self.region_selection: PolygonInteractor = None
        self.point_selection_enabled = False

    @exception_handled
    def create_selection_panel(self):
        self.selection_label = ipw.Label(value='Selection Operation:')
        self.selection = ipw.RadioButtons(  options=['spectral graph', 'select point', 'select region'], disabled=False, layout={'width': 'max-content'} )
        self.selection.observe( self.set_selection_mode, "value" )

    def set_selection_mode( self, event: Dict ):
        smode = event['new']
        rselection = (smode == 'select region')
        self.enable_region_selection( rselection )
        self.point_selection_enabled = (smode == 'select point')

    def enable_region_selection(self, enabled: bool ):
        self.region_selection.set_enabled( enabled )

    def set_region_color(self, color: str ):
        self.region_selection.set_color(color)

    def add_slider(self,  **kwargs ):
        if self.slider is None:
            self.slider = PageSlider( self.slider_axes, self.nFrames )
            self.slider_cid = self.slider.on_changed(self._update)

    @exception_handled
    def _update( self, val ):
        if self.slider is not None:
            tval = self.slider.val
            self.currentFrame = int( tval )
            lgm().log(f"Slider Update, frame = {self.currentFrame}")
#            ufm().show( f"Loading frame {self.currentFrame}", "yellow" )
            self.update_plots()
#            ufm().clear()

    @exception_handled
    def update_plots(self):
        if self.image is not None:
            fdata: xa.DataArray = self.frame_data
            if fdata is not None:
                self.image.set_data(fdata.values)
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
            self.nFrames = self.data.shape[0]
            self.band_axis = kwargs.pop('band', 0)
            self.z_axis_name = self.data.dims[self.band_axis]
            self.x_axis = kwargs.pop('x', 2)
            self.x_axis_name = self.data.dims[self.x_axis]
            self.y_axis = kwargs.pop('y', 1)
            self.y_axis_name = self.data.dims[self.y_axis]

    def gui(self,**kwargs):
        self.setBlock()
        self.base = TileServiceBasemap()
        [x0, x1, y0, y1] = self.block.extent()
        self.base.setup_plot( (x0,x1), (y0,y1), **kwargs )
        self.init_map(**kwargs)
        self.region_selection = PolygonInteractor( self.base.gax )
        return self.base.gax.figure.canvas

    def init_map(self,**kwargs):
        self.image: AxesImage = self.frame_data.plot.imshow( ax=self.base.gax, alpha=0.3)
        self.add_slider(**kwargs)

if __name__ == '__main__':
    dm: DataManager = DataManager.initialize("demo2", 'desis')
    dm.loadCurrentProject("main")
    classes = [('Class-1', "cyan"), ('Class-2', "green"), ('Class-3', "magenta"), ('Class-4', "blue")]
    lm().setLabels(classes)

    mm = MapManager()
    panel = mm.gui()
    plt.show( )


