import os
import conda, numbers

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
from affine import Affine
from matplotlib.backend_bases import MouseEvent
from matplotlib.image import AxesImage
from affine import Affine
import numpy as np
from matplotlib.axes import Axes
from scipy import stats as sps
import matplotlib.pyplot as plt
from spectraclass.gui.spatial.widgets.tiles import TileSelector
import cartopy.crs as ccrs
from spectraclass.gui.spatial.widgets.crs import get_ccrs
from typing import List, Union, Tuple, Optional, Dict, Iterable

import pyproj.crs as pcrs
import rasterio as rio
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker

def to_proj4( crs: str ) -> Dict:
    tups = [ kv.split('=') for kv in crs.split('+') ]
    return { t[0].strip(): t[1].strip() for t in tups if (len(t) == 2) }

def idx_of_nearest( array: np.ndarray, value: float ):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

class Tile:

    def __init__(self, data: xa.DataArray, tile_size: int ):
        assert data.ndim in [2,3], f"Can't defin Tile with {data.ndim} dims"
        self._data: xa.DataArray = data.expand_dims( {"band":1}, 0 ) if data.ndim == 2 else data
        self._xc: np.ndarray = data.coords[ data.dims[2] ].data
        self._yc: np.ndarray = data.coords[ data.dims[1] ].data
        self._tile_size: int = tile_size

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

    def get_full_extent(self):
        dx, dy = (self._xc[-1]-self._xc[-2]), (self._yc[-1]-self._yc[-2])
        return self._xc[0], self._xc[-1]+dx, self._yc[0], self._yc[-1]+dy

    def get_tile(self, dloc: List[float] = None, origin: str = "upper" ) -> xa.DataArray:
        if dloc is None:
            loc = [ self._data.shape[2]//2, self._data.shape[1]//2 ]
            dloc = [ self._xc[loc[0]], self._yc[loc[1]],  ]
        else:
            loc = [ idx_of_nearest(self._xc,dloc[0]), idx_of_nearest(self._yc,dloc[1]) ]
        iext = [ loc[i]+self._tile_size for i in (0,1)  ]
        tile = self._data[ :, loc[1]:iext[1], loc[0]:iext[0] ]
        transform: List = list(self._data.attrs['transform'])
        transform[2], transform[5] = dloc[0], dloc[1]
        tile.attrs['transform'] = transform
        tile.attrs['extent'] = self.extent( transform, tile.shape, origin )
        return tile

def fill_nodata( da: xa.DataArray, fill_val ) -> xa.DataArray:
    nodata = da.attrs.get('nodatavals')
    if nodata is not None:
        filled_data: np.ndarray = np.where( da.data == nodata, fill_val, da.data )
        return da.copy( data=filled_data )
    else: return da

def downscale( geo_xarray: xa.DataArray, block_size: int, origin: str ) -> xa.DataArray:
    nbands, ishp = geo_xarray.shape[0], geo_xarray.shape[1:]
    new_shape = ishp[0]//block_size, ishp[1]//block_size
    nsh = [ ishp[0]//new_shape[0], ishp[1]//new_shape[1] ]
    sh = nbands, new_shape[0], nsh[0], new_shape[1], nsh[1]
    sxa: xa.DataArray = geo_xarray[ :, :new_shape[0]*nsh[0], :new_shape[1]*nsh[1] ]
    ta: np.ndarray = sxa.data.reshape(sh)
    tas: np.ndarray = ta.transpose( (0,1,3,2,4) ).reshape( (nbands, new_shape[0], new_shape[1], nsh[0]*nsh[1]) )
    if issubclass( geo_xarray.dtype.type, numbers.Integral ):
        modes, counts = sps.mode( tas, axis=3, nan_policy='omit' )
    else: modes = tas.mean( axis=3 )
    dxa: xa.DataArray = sxa[ :, ::block_size, ::block_size ]
    dxa = dxa.copy( data = modes.reshape( dxa.shape ) )
    transform: List = list(geo_xarray.transform)
    for i in [0,1,3,4]: transform[i] = transform[i]*block_size
    dxa.attrs['transform'] = transform
    dxa.attrs['extent'] = Tile.extent( transform, new_shape, origin )
    return dxa

iband = 0
block_size = 100
origin = "upper"
blocks_per_tile = 5
tile_size = block_size*blocks_per_tile
LabelDataFile = "/Users/tpmaxwel/GDrive/Tom/Data/ChesapeakeLandUse/CalvertCounty/CALV_24009_LandUse.tif"

da: xa.DataArray = fill_nodata( xa.open_rasterio( LabelDataFile ), -1 )
downscaled_data: xa.DataArray = downscale( da, block_size, origin )
tile = Tile( da, tile_size )

proj4_attrs: Dict = to_proj4( da.attrs["crs"] )
proj_crs = pcrs.CRS( proj4_attrs )
cart_crs: ccrs.CRS = get_ccrs( proj4_attrs )

fig = plt.figure( figsize=(16,8) )
ax0: Axes = fig.add_subplot( 121, projection=cart_crs )
tile_array0 = tile.get_tile( None, origin )
tile_data = tile_array0.data[iband]
vmin, vmax = da.data.min(), da.data.max(),
img0: AxesImage = ax0.imshow( tile_data, transform=cart_crs, origin=origin, cmap="tab20", extent=tile_array0.attrs["extent"], vmin=vmin, vmax=vmax )

def on_tile_selection( event: MouseEvent ):
    dloc = [ event.xdata, event.ydata ]
    tile_array = tile.get_tile( dloc, origin )
    tile_ext = tile_array.attrs["extent"]
    img0.set_extent( tile_ext )
    tile_data = tile_array.data[iband]
    print(f"on_tile_selection: loc = {dloc}, dshape={tile_data.shape}, extent = {tile_ext}, vrange = [ {tile_data.min()}, {tile_data.max()} ]")
    img0.set_data( tile_data )
    img0.figure.canvas.draw()
    img0.figure.canvas.flush_events()

ax1 = fig.add_subplot( 122   )
img1: AxesImage = ax1.imshow( downscaled_data.data[iband], origin='upper', cmap="tab20", vmin=vmin, vmax=vmax )
img1.set_extent( downscaled_data.extent )
rsize = Tile.dsize( downscaled_data.transform, blocks_per_tile )
ts = TileSelector( ax1, rsize, on_tile_selection )
ts.activate()
plt.show()

# from mpl_toolkits.basemap import Basemap
# def to_basemap( proj4: Dict, da: xa.DataArray, range: Tuple ) -> Dict:
#     bmap = { k:v for k,v in proj4.items() if k[:4] in ( "lat_", "lon_" ) }
#     bmap['projection'] =   proj4['proj']
#     x: xa.IndexVariable = da.coords[da.dims[2]]
#     y: xa.IndexVariable = da.coords[da.dims[1]]
#     if proj4['units'] == "m":
#         bmap['llcrnrx'] = x.data[range[0]]
#         bmap['llcrnry'] = y.data[range[1]]
#         bmap['urcrnrx'] = x.data[range[2]]
#         bmap['urcrnry'] = y.data[range[3]]
#     else:
#         bmap['llcrnrlon'] = x.data[range[0]]
#         bmap['llcrnrlat'] = y.data[range[1]]
#         bmap['urcrnrlon'] = x.data[range[2]]
#         bmap['urcrnrlat'] = y.data[range[3]]
#     return bmap
# bmap_args = to_basemap( proj4_attrs, da, range )
# bmap = Basemap( ax=ax, **bmap_args )
# bmap.imshow( da.data[0] )