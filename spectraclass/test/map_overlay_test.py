import os
import conda, numbers

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
import rioxarray as rio
from affine import Affine
from matplotlib.backend_bases import MouseEvent
from matplotlib.image import AxesImage
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from spectraclass.gui.spatial.widgets.tiles import TileSelector, TileManager
import cartopy.crs as ccrs
from spectraclass.data.base import DataManager, dm

from typing import List, Union, Tuple, Optional, Dict, Iterable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
from spectraclass.data.spatial.tile.manager import  tm

iband = 0
block_size = 100
origin = "upper"
blocks_per_tile = 5
tile_size = block_size*blocks_per_tile
# LabelDataFile = "/Users/tpmaxwel/GDrive/Tom/Data/ChesapeakeLandUse/CalvertCounty/CALV_24009_LandUse.tif"
LabelDataFile = "/Users/tpmaxwel/GDrive/Tom/Data/ChesapeakeLandUse/AnneArundelCounty/ANNE_24003_LandUse.tif"

SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_002-20200628T153803-V0210-SPECTRAL_IMAGE.tif"

tmgr = TileManager( LabelDataFile, tile_size, origin, nodata_fill=-1 )
spectral_array: xa.DataArray = TileManager.read_data_layer( SpectralDataFile, origin )
crs: ccrs.CRS = tmgr.crs()

fig = plt.figure( figsize=(16,8) )
ax0: Axes = fig.add_subplot( 121, projection=crs )
tile_array: xa.DataArray = tmgr.get_tile( None, origin )
tile_extent = tile_array.attrs["extent"]
tile_data = tile_array.data[iband]

vr = tmgr.vrange()
img0: AxesImage = ax0.imshow( tile_data, transform=crs, origin=origin, cmap="tab20", extent=tile_extent, vmin=vr[0], vmax=vr[1] )

crs1 = spectral_array.attrs['ccrs']
(x0,y0) = crs1.transform_point(tile_extent[0], tile_extent[2], crs)
(x1,y1) = crs1.transform_point(tile_extent[1], tile_extent[3], crs)
spectral_array_extent = TileManager.extent( spectral_array.attrs['transform'], spectral_array.shape, origin )
spectral_tile = spectral_array.loc[:,y0:y1,x0:x1]
spectral_tile_transform = TileManager.clipped_transform( spectral_tile, (x0, y0) )
spectral_tile.attrs['transform'] = spectral_tile_transform
spectral_tile_extent = TileManager.extent( spectral_tile_transform, spectral_tile.shape, origin )
spectral_tile.attrs['extent'] = spectral_tile_extent
img1: AxesImage = ax0.imshow( spectral_tile.data[iband], transform=crs1, origin=origin, cmap="tab20", extent=spectral_tile_extent )

plt.show()
