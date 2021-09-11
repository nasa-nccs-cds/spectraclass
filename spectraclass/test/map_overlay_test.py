import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
from affine import Affine
from matplotlib.backend_bases import MouseEvent
from matplotlib.image import AxesImage
import numpy as np
from matplotlib.axes import Axes
from scipy import stats as sps
import matplotlib.pyplot as plt
from spectraclass.gui.spatial.widgets.tiles import TileSelector
import cartopy.crs as ccrs
from spectraclass.gui.spatial.widgets.crs import get_ccrs
from typing import List, Union, Tuple, Optional, Dict

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

def downscale( a: np.ndarray, block_size: int, nodata=None ) -> np.ndarray:
    new_shape = a.shape[0]//block_size, a.shape[1]//block_size
    nsh = [ a.shape[0]//new_shape[0], a.shape[1]//new_shape[1] ]
    sh = new_shape[0], nsh[0], new_shape[1], nsh[1]
    ta: np.ndarray = a[ :new_shape[0]*nsh[0], :new_shape[1]*nsh[1] ].reshape(sh)
    tas: np.ndarray = ta.transpose( (0,2,1,3) ).reshape( (new_shape[0], new_shape[1], nsh[0]*nsh[1]) )
    modes, counts = sps.mode( tas, 2, nan_policy='omit' )
    return np.squeeze( modes if nodata is None else np.where( modes==nodata, -1, modes ) )

class Tile:

    def __init__(self, data: xa.DataArray, tile_size: int ):
        self._data: xa.DataArray = data
        self._xc: np.ndarray = data.coords[ data.dims[1] ].data
        self._yc: np.ndarray = data.coords[ data.dims[0] ].data
        self._tile_size: int = tile_size

    def get_full_extent(self):
        dx, dy = (self._xc[-1]-self._xc[-2]), (self._yc[-1]-self._yc[-2])
        return self._xc[0], self._xc[-1]+dx, self._yc[0], self._yc[-1]+dy

    def get_tile(self, dloc: List[float] = None ) -> xa.DataArray:
        if dloc is None:    loc = [ self._data.shape[1]//2, self._data.shape[0]//2 ]
        else:               loc = [ idx_of_nearest(self._xc,dloc[0]), idx_of_nearest(self._yc,dloc[1]) ]
        iext = [ loc[i]+self._tile_size for i in (0,1)  ]
        cext = [ self._xc[ loc[0] ], self._xc[ loc[0]+self._tile_size ],
                self._yc[ loc[1] ], self._yc[ loc[1]+self._tile_size ] ]
        tile = self._data[ loc[1]:iext[1], loc[0]:iext[0] ]
        tile.attrs["extent"] = cext
        return tile

block_size = 100
blocks_per_tile = 5
tile_size = block_size*blocks_per_tile
LabelDataFile = "/Users/tpmaxwel/GDrive/Tom/Data/ChesapeakeLandUse/CalvertCounty/CALV_24009_LandUse.tif"

da: xa.DataArray = xa.open_rasterio( LabelDataFile ).squeeze(drop=True)
downscaled_data: np.ndarray = downscale( da.data, block_size, da.nodatavals )
tile = Tile( da, tile_size )

proj4_attrs: Dict = to_proj4( da.attrs["crs"] )
proj_crs = pcrs.CRS( proj4_attrs )
cart_crs: ccrs.CRS = get_ccrs( proj4_attrs )

fig = plt.figure( figsize=(16,8) )
ax0: Axes = fig.add_subplot( 121, projection=cart_crs )
tile_array0 = tile.get_tile()
img0: AxesImage = ax0.imshow( tile_array0.data, transform=cart_crs, origin='upper', cmap="tab20", extent=tile_array0.attrs["extent"] )

def on_tile_selection( event: MouseEvent ):
    dloc = [ event.xdata, event.ydata ]
    print( f"on_tile_selection: loc = {dloc}" )
    tile_array = tile.get_tile( dloc )
    tile_ext = tile_array.attrs["extent"]
    img0.set_extent( tile_ext )
    img0.set_data( tile_array.data )
    img0.figure.canvas.draw()

ax1 = fig.add_subplot( 122   )
img1: AxesImage = ax1.imshow( downscaled_data, origin='upper', cmap="tab20" )
img1.set_extent( tile.get_full_extent() )
ts = TileSelector( ax1, blocks_per_tile, on_tile_selection )
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