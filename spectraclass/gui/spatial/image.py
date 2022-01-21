from matplotlib.patches import Rectangle
import xarray as xa
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import PickEvent, MouseEvent
import contextlib, time
from typing import List, Optional, Dict, Tuple
from matplotlib.image import AxesImage
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.gui.spatial.source import WMTSRasterSource
import matplotlib.artist

def toXA( vname: str, nparray: np.ndarray, format="np", transpose = False ):
    gs: List[int] = [*nparray.shape]
    if nparray.ndim == 2:
        dims = ['y', 'x']
        coords = { dims[i]: np.array(range(gs[i])) for i in (0, 1) }
    elif nparray.ndim == 3:
        if transpose:
            nparray = nparray.reshape([gs[0] * gs[1], gs[2]]).transpose().reshape([gs[2], gs[0], gs[1]])
        dims = ['band', 'y', 'x']
        coords = { dims[i]: np.array(range(nparray.shape[i])) for i in (0, 1, 2) }
    else:
        raise Exception(f"Can't convert numpy->xa array with {nparray.ndim} dims")
    return xa.DataArray( nparray, coords, dims, vname, dict(transform=[1, 0, 0, 0, 1, 0], fileformat=format))


class TileServiceImage(AxesImage):

    def __init__(self, ax: Axes, raster_source: WMTSRasterSource, projection, **kwargs):
        self.raster_source: WMTSRasterSource = raster_source
        xrange = kwargs.pop('xrange',None)
        yrange = kwargs.pop('yrange', None)
        kwargs.setdefault('in_layout', False)
        block_selection = kwargs.pop('block_selection', False)
        self.user_is_interacting = False
        super().__init__(ax, **kwargs)
        self.projection = projection
        self.cache = []
        self.current_extent = []
        self._block_selection_callback = None
        self._selected_block: Rectangle = None
        self._blocks: List[Rectangle] = []
        self.axes.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.axes.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.axes.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.on_release()
        if xrange is not None: self.axes.set_xbound( xrange[0], xrange[1] )
        if yrange is not None: self.axes.set_ybound( yrange[0], yrange[1] )

        with self.hold_limits():
            self.axes.add_image( self )

        if block_selection:
            self.add_block_selection( )

    def set_bounds(self, xrange: Tuple[float], yrange: Tuple[float] ):
        if xrange is not None: self.axes.set_xbound( xrange[0], xrange[1] )
        if yrange is not None: self.axes.set_ybound( yrange[0], yrange[1] )

    def update_blocks(self):
        self.clear_blocks()
        self.add_block_selection()

    def clear_blocks(self):
        for r in self._blocks: r.remove()
        self._blocks = []

    def add_block_selection(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        mdata = tm().tile_metadata
        [dx, _, x0, _, dy, y0 ] = tm().transform
        block_shape = tm().block_shape
        block_size = mdata['block_size']
        width, height = dx*block_shape[0], dy*block_shape[1]
        maxsize = max( block_size.values() )
        for bc, bsize in block_size.items():
            if bsize > 0:
                selected = (list(bc) == tm().block_index)
                xc, yc = x0 + width*bc[0], y0+height*bc[1]
                lw = 3 if selected else 1
                color = 'orange' if selected else 'yellow'
                r = Rectangle( (xc,yc), width, height, fill=False, edgecolor=color, lw=lw, alpha=bsize/maxsize )
                setattr( r, 'block_index', bc )
                r.set_picker( True )
                self.axes.add_patch( r )
                self._blocks.append( r )
                if selected: self._selected_block = r
         #       lgm().log( f" BLOCK[{bc}]: xc={xc:.1f}, yc={yc:.1f}, size=({width:.1f},{height:.1f})\n  ->> Axis bounds: xlim={self.axes.get_xlim()}, ylim={self.axes.get_ylim()}", print=True )

    def on_press(self, event: MouseEvent =None):
        self.user_is_interacting = True
        lgm().log( f" Button Press: {(event.xdata,event.ydata)}")

    def on_release(self, event=None):
        self.user_is_interacting = False
        self.stale = True

    def select_block(self, r: Rectangle ):
        from spectraclass.gui.spatial.map import MapManager, mm
        if r != self._selected_block:
            lgm().log(f"\n  ******** Selected block: {r.block_index}  ******** ")
            if self._selected_block is not None:
                self._selected_block.set_linewidth(1)
                self._selected_block.set_color("yellow")
            r.set_linewidth(3)
            r.set_color("orange")
            self._selected_block = r
            self.figure.canvas.draw_idle()
            mm().setBlock( r.block_index, update=True )

    def on_pick(self, event: PickEvent =None):
        lgm().log( f" Pick Event: type = {type(event)}" )
        if type(event.artist) == Rectangle:
            self.select_block( event.artist )

    def get_window_extent(self, renderer=None):
        return self.axes.get_window_extent(renderer=renderer)

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible():
            return
        window_extent = self.axes.get_window_extent()
        [x1, y1], [x2, y2] = self.axes.viewLim.get_points()
        extent_changed = ( self.current_extent != [x1, y1, x2, y2] )
        if (not self.user_is_interacting) and extent_changed:
            located_images = self.raster_source.fetch_raster( self.projection, extent=[x1, x2, y1, y2], target_resolution=(window_extent.width, window_extent.height))
            self.cache = located_images
            self.current_extent = [x1, y1, x2, y2]

        for img, extent in self.cache:
            self.set_array(img)
            with self.hold_limits():
                self.set_extent(extent)
            super().draw(renderer, *args, **kwargs)

    def can_composite(self):
        return False

    @contextlib.contextmanager
    def hold_limits( self ):
        data_lim = self.axes.dataLim.frozen().get_points()
        view_lim = self.axes.viewLim.frozen().get_points()
        other = (self.axes.ignore_existing_data_limits, self.axes._autoscaleXon, self.axes._autoscaleYon)
        try:
            yield
        finally:
            self.axes.dataLim.set_points(data_lim)
            self.axes.viewLim.set_points(view_lim)
            (self.axes.ignore_existing_data_limits, self.axes._autoscaleXon, self.axes._autoscaleYon) = other
