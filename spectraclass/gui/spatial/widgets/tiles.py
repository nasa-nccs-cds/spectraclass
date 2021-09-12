import numpy as np
import sys, numbers
from copy import deepcopy
from spectraclass.gui.spatial.widgets.crs import get_ccrs
import cartopy.crs as ccrs
import pyproj.crs as pcrs
from scipy import stats as sps
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict, Callable
from matplotlib.backend_bases import MouseEvent
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class TileSelector:
    INIT_POS = ( sys.float_info.max, sys.float_info.max )

    def __init__( self, ax: Axes, rsize: Tuple[float,float], on_click: Callable[[MouseEvent],None] ):
        self._rsize = rsize
        self._ax = ax
        self.rect = self._rect( self.INIT_POS )
        self._on_click = on_click
        self._background = None
        self._selection_background = None
        self._selection_rect = None
        self._active = False
        self.canvas = None

    def _rect( self, pos: Tuple[float,float] ):
        rect = Rectangle( pos, self._rsize[0], self._rsize[1], facecolor="white", edgecolor="black", alpha=1.0 )
        self._ax.add_patch(rect)
        return rect

    def activate(self):
        self._active = True
        self.canvas = self.rect.figure.canvas
        self.cidpress   = self.canvas.mpl_connect( 'button_press_event', self.on_mouse_click )
        self.cidmotion  = self.canvas.mpl_connect( 'motion_notify_event', self.on_motion )
        self.rect.set_animated(True)
        self.rect.set_visible(True)

    def on_mouse_click( self, event: MouseEvent ):
        if self._selection_background is not None:
            self.canvas.restore_region(self._selection_background )
        self._selection_background = self.canvas.copy_from_bbox( self.rect.axes.bbox )
        if self._selection_rect is None:
            self._selection_rect = self._rect( (event.xdata,event.ydata) )
        else:
            self._selection_rect.set_x( event.xdata )
            self._selection_rect.set_y( event.ydata )
        self._ax.draw_artist(self.rect)
        self._on_click( event )

    def on_motion(self, event: MouseEvent ):
        if (event.inaxes == self.rect.axes):
            self.rect.set_x( event.xdata )
            self.rect.set_y( event.ydata )
            if self._background is not None:
                self.canvas.restore_region(self._background)
            self._background = self.canvas.copy_from_bbox(self.rect.axes.bbox)
            self._ax.draw_artist(self.rect)
            if self._selection_rect is not None:
                self._ax.draw_artist( self._selection_rect )
            self.canvas.draw()
        else:
            if self._background is not None:
                self.canvas.restore_region(self._background)
                self._background = None
                self.canvas.draw()

    def deactivate(self):
        self._active = False
        if self._selection_rect is not None:
            self._selection_rect.set.set_visible( False )
            self._selection_rect = None
        if self._background is not None:
            self.canvas.restore_region( self._background )
            self._background = None
        if self._selection_background is not None:
            self.canvas.restore_region( self._selection_background )
            self._selection_background = None
        self.rect.set_x( self.INIT_POS[0] )
        self.rect.set_y( self.INIT_POS[1] )
        self.canvas.mpl_disconnect( self.cidpress )
        self.canvas.mpl_disconnect( self.cidmotion )
        self.rect.set_animated( False )
        self.rect.set_visible( False )
        self.canvas.draw()
        self.canvas = None

class TileManager:

    def __init__(self, file_path: str, tile_size: int, **kwargs  ):
        data = xa.open_rasterio( file_path )
        assert data.ndim in [2, 3], f"Can't defin Tile with {data.ndim} dims"
        self._tile_size = tile_size
        nodata_fill = kwargs.get( 'nodata_fill', None )
        if nodata_fill is not None:
            data = self.fill_nodata( data, nodata_fill )
        self._data: xa.DataArray = data.expand_dims( {"band":1}, 0 ) if data.ndim == 2 else data
        self._xc: np.ndarray = data.coords[ data.dims[2] ].data
        self._yc: np.ndarray = data.coords[ data.dims[1] ].data

    def vrange(self) -> Tuple[float,float]:
        return ( self._data.data.min(), self._data.data.max() )

    def crs(self) -> ccrs.CRS:
        proj4_attrs: Dict = self.to_proj4( self._data.attrs["crs"] )
        return get_ccrs(proj4_attrs)

    @staticmethod
    def to_proj4( crs: str ) -> Dict:
        tups = [ kv.split('=') for kv in crs.split('+') ]
        return { t[0].strip(): t[1].strip() for t in tups if (len(t) == 2) }

    @staticmethod
    def fill_nodata( da: xa.DataArray, fill_val ) -> xa.DataArray:
        nodata = da.attrs.get('nodatavals')
        if nodata is not None:
            filled_data: np.ndarray = np.where( da.data == nodata, fill_val, da.data )
            return da.copy( data=filled_data )
        else: return da

    def downscale( self, block_size: int, origin: str ) -> xa.DataArray:
        nbands, ishp = self._data.shape[0], self._data.shape[1:]
        new_shape = ishp[0]//block_size, ishp[1]//block_size
        nsh = [ ishp[0]//new_shape[0], ishp[1]//new_shape[1] ]
        sh = nbands, new_shape[0], nsh[0], new_shape[1], nsh[1]
        sxa: xa.DataArray = self._data[ :, :new_shape[0]*nsh[0], :new_shape[1]*nsh[1] ]
        ta: np.ndarray = sxa.data.reshape(sh)
        tas: np.ndarray = ta.transpose( (0,1,3,2,4) ).reshape( (nbands, new_shape[0], new_shape[1], nsh[0]*nsh[1]) )
        if issubclass( self._data.dtype.type, numbers.Integral ):
            modes, counts = sps.mode( tas, axis=3, nan_policy='omit' )
        else: modes = tas.mean( axis=3 )
        dxa: xa.DataArray = sxa[ :, ::block_size, ::block_size ]
        dxa = dxa.copy( data = modes.reshape( dxa.shape ) )
        transform: List = list(self._data.transform)
        for i in [0,1,3,4]: transform[i] = transform[i]*block_size
        dxa.attrs['transform'] = transform
        dxa.attrs['extent'] = self.extent( transform, new_shape, origin )
        return dxa

    @staticmethod
    def idx_of_nearest(array: np.ndarray, value: float):
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

    @staticmethod
    def dsize( transform: Union[List, Tuple], isize: int ) -> Tuple[float,float]:
        return (  abs( isize * transform[0] + isize * transform[1] ),  abs( isize * transform[3] + isize * transform[4] ) )

    @staticmethod
    def extent( transform: Union[List, Tuple], shape: Union[List, Tuple], origin: str ):
        (sy,sx) = (shape[1],shape[2]) if len(shape) == 3 else (shape[0],shape[1])
        ext =  [transform[2], transform[2] + sx * transform[0] + sy * transform[1],
                transform[5], transform[5] + sx * transform[3] + sy * transform[4]]
        if origin == "upper": ( ext[2], ext[3] ) = ( ext[3], ext[2] )
        return ext

    def get_tile(self, dloc: List[float] = None, origin: str = "upper" ) -> xa.DataArray:
        if dloc is None:
            loc = [ self._data.shape[2]//2, self._data.shape[1]//2 ]
            dloc = [ self._xc[loc[0]], self._yc[loc[1]],  ]
        else:
            loc = [ self.idx_of_nearest(self._xc,dloc[0]), self.idx_of_nearest(self._yc,dloc[1]) ]
        if origin == "upper":  tile = self._data[:, loc[1]-self._tile_size:loc[1], loc[0]:loc[0]+self._tile_size ]
        else:                  tile = self._data[:, loc[1]:loc[1]+self._tile_size, loc[0]:loc[0]+self._tile_size ]
        transform: List = list(self._data.attrs['transform'])
        transform[2], transform[5] = dloc[0], dloc[1]
        tile.attrs['transform'] = transform
        tile.attrs['extent'] = self.extent( transform, tile.shape, origin )
        return tile



if __name__ == '__main__':

    def on_selection( event: MouseEvent ):
        print( f"Location Selection: {event}")

    fig: plt.Figure = plt.figure()
    ax = fig.add_subplot( 111 )
    ax.set_xlim( -100.0, 100.0 )
    ax.set_ylim( -100.0, 100.0 )
    blocks_per_tile = 5
    dr = TileSelector( ax, blocks_per_tile, on_selection )
    dr.activate()
    plt.show()