import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
from affine import Affine
import numpy as np
import matplotlib.pyplot as plt
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

def downscale( a: np.ndarray, factor: int ):
    new_shape = a.shape[0]//factor, a.shape[1]//factor
    nsh = [ a.shape[0]//new_shape[0], a.shape[1]//new_shape[1] ]
    ta = a[ :new_shape[0]*nsh[0], :new_shape[1]*nsh[1] ]
    sh = new_shape[0], nsh[0], new_shape[1], nsh[1]
    return ta.reshape(sh).min(-1).min(1)

origin = [ 15100, 31000 ]
size = [ 500, 500 ]
xr = (  origin[0], origin[0]+size[0] )
yr = (  origin[1], origin[1]+size[1] )

LabelDataFile = "/Users/tpmaxwel/GDrive/Tom/Data/ChesapeakeLandUse/CalvertCounty/CALV_24009_LandUse.tif"

full_da: xa.DataArray = xa.open_rasterio( LabelDataFile )
downscaled_data: np.ndarray = downscale( full_da.data[0], 100 )
da: xa.DataArray = full_da[0, yr[0]:yr[1], xr[0]:xr[1]]
transform: Affine = Affine.from_gdal( *da.attrs["transform"] )
proj4_attrs: Dict = to_proj4( da.attrs["crs"] )
# rio_crs = rcrs.CRS.from_proj4( da.attrs["crs"] )
proj_crs = pcrs.CRS( proj4_attrs )
cart_crs: ccrs.CRS = get_ccrs( proj4_attrs )

fig = plt.figure( figsize=(16,8) )
ax0 = fig.add_subplot( 121, projection=cart_crs )
img0 = ax0.imshow( da.data, transform=cart_crs, origin='upper', cmap="tab20" )

ax1 = fig.add_subplot( 122   )
ax1.imshow( downscaled_data, origin='upper' )
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