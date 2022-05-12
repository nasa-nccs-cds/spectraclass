from spectraclass.data.base import DataManager
from matplotlib.image import AxesImage
from matplotlib.backend_bases import PickEvent, MouseEvent
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, RegularPolygon, Polygon
from spectraclass.gui.control import UserFeedbackManager, ufm
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from cartopy.mpl.geoaxes import GeoAxes
from spectraclass.xext.xgeo import XGeo
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.spatial.tile.manager import TileManager, tm
import matplotlib.pyplot as plt
import ipywidgets as ipw
import time, xarray as xa

class AvirisDatasetManager:

    @log_timing
    def __init__(self, **kwargs):
        self.dm: DataManager = DataManager.initialize( "DatasetManager", 'aviris' )
        if "cache_dir" in kwargs: self.dm.modal.cache_dir = kwargs["cache_dir"]
        if "data_dir"  in kwargs: self.dm.modal.data_dir = kwargs["data_dir"]
        if "images_glob" in kwargs: self.dm.modal.images_glob = kwargs["images_glob"]
        self.dm.modal.ext = kwargs.get( "ext", "_img" )
        self.init_band = kwargs.get( "init_band", 160 )
        self.grid_color = kwargs.get("grid_color", 'white')
        self.selection_color = kwargs.get("selection_color", 'black')
        self.grid_alpha = kwargs.get("grid_alpha", 0.5 )
        self.slw = kwargs.get("slw", 3)
        self.dm.proc_type = "cpu"
        self._blocks: List[Rectangle] = []
        self._selected_block = None
        TileManager.block_size = kwargs.get( 'block_size',  250 )
        self.nimages = len( self.dm.modal.image_names )
        self._nbands = None
        self.band_selector: ipw.IntSlider = None
        self.band_plot: AxesImage = None
        self._axes: Axes = None
        lgm().log( f"AvirisDatasetManager: Found {self.nimages} images "  )

    @property
    def nbands(self) -> int:
        if self._nbands is None:
            data_array: xa.DataArray = tm().tile.data
            self._nbands = data_array.shape[0]
        return self._nbands

    @property
    def band_index(self) -> int:
        if self.band_selector is None:
            return self.init_band
        else:
            return self.band_selector.value

    @property
    def image_index(self) -> int:
        return self.dm.modal.file_selector.index

    @property
    def image_name(self) -> str:
        return self.dm.modal.get_image_name( self.image_index )

    def update_image( self ):
        self.dm.modal.set_current_image( self.image_index )
        data_array: xa.DataArray = tm().tile.data
        band_array: np.ndarray = data_array[self.band_index].values.squeeze().transpose()
        nodata = data_array.attrs.get('_FillValue')
        band_array[band_array == nodata] = np.nan
        vmean, vstd = np.nanmean(band_array), np.nanstd( band_array )
        vrange = [ max(vmean-2*vstd, 0.0), vmean+2*vstd ]
        if self.band_plot is None:  self.band_plot = self.ax.imshow( band_array, cmap="jet")
        else:                       self.band_plot.set_data( band_array )
        self.band_plot.set_clim(*vrange)

    def on_image_change( self, event: Dict ):
        ufm().show( f"Loading image {self.image_name}" )
        self.update_image()

    def on_band_change( self, event: Dict ):
        ufm().show( f"Plotting band {self.band_index}" )
        self.update_image()

    def gui(self):
        plt.ioff()
        self.update_image()
        self.add_block_selection()
        self.band_selector = ipw.IntSlider( self.init_band, 0, self.nbands, 1 )
        self.band_selector.observe( self.on_band_change, "value" )
        self.dm.modal.set_file_selection_observer( self.on_image_change )
        control_panel = ipw.VBox( [ufm().gui(), self.dm.modal.file_selector, self.band_selector, self.band_plot.figure.canvas] )
        widget = ipw.HBox( [ control_panel], layout=ipw.Layout(flex='1 1 auto') )
        plt.ion()
        return widget

    @property
    def ax(self) -> Axes:
        if self._axes is None:
            self._axes: Axes = plt.axes()
            self._axes.set_yticks([])
            self._axes.set_xticks([])
            self._axes.figure.canvas.mpl_connect( 'pick_event', self.on_pick )
        return self._axes

    def xc(self, ix: int, iy: int ) -> float:
        t = tm().tile.data.attrs['transform']
        return t[2] + t[0]*ix + t[1]*iy

    def dx(self) -> float:
        t = tm().tile.data.attrs['transform']
        return  t[0]

    def yc(self, ix: int, iy: int ) -> float:
        t = tm().tile.data.attrs['transform']
        return t[5] + t[3]*ix + t[4]*iy

    def dy(self) -> float:
        t = tm().tile.data.attrs['transform']
        return  t[3]

    def pc(self, ix: int, iy: int) -> Tuple[float,float]:
        t = tm().tile.data.attrs['transform']
        return ( t[2] + t[0] * ix + t[1] * iy, t[5] + t[3]*ix + t[4]*iy )

    def patch(self, ix: int, iy: int, size: int ) -> np.ndarray:
        coords = [ self.pc(ix,iy), self.pc(ix+size,iy), self.pc(ix+size,iy+size), self.pc(ix,iy+size) ]
        return np.array( coords )

    @exception_handled
    def add_block_selection(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        transform = tm().tile.data.attrs['transform']
        block_size = tm().block_size
        block_dims = tm().block_dims
        lgm().log(f"  add_block_selection: block_size={block_size}, block_dims={block_dims}, transform={transform} ")
        for tx in range( block_dims[0] ):
            for ty in range( block_dims[1] ):
                selected = ( (tx,ty) == tm().block_index )
                ix, iy = tx*block_size, ty*block_size
                lw = self.slw if ( selected ) else 1
                color = self.selection_color if ( selected ) else self.grid_color
                r = Rectangle( (iy, ix), block_size, block_size, fill=False, edgecolor=color, lw=lw, alpha=self.grid_alpha )
                setattr( r, 'block_index', (tx,ty) )
                r.set_picker( True )
                self.ax.add_patch( r )
                self._blocks.append( r )
                if selected: self._selected_block = r

    def select_block(self, r: Rectangle ):
        if r != self._selected_block:
            lgm().log(f"  ******** Selected block: {r.block_index}  ******** ")
            if self._selected_block is not None:
                self._selected_block.set_color( self.grid_color )
            r.set_linewidth(self.slw)
            r.set_color( self.selection_color )
            self._selected_block = r
            self.ax.figure.canvas.draw_idle()
 #           mm().setBlock( r.block_index, update=True )

    def on_pick(self, event: PickEvent =None):
        lgm().log( f" Pick Event: type = {type(event)}" )
        if type(event.artist) == Rectangle:
            self.select_block( event.artist )
