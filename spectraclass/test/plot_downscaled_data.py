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

iband = 0
block_size = 4
origin = "upper"
cmap="jet"
blocks_per_tile = 5
tile_size = block_size*blocks_per_tile
SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_002-20200628T153803-V0210-SPECTRAL_IMAGE.tif"

tmgr = TileManager( SpectralDataFile, tile_size, origin, nodata_fill=-1 )
downscaled: xa.DataArray = tmgr.downscale( block_size, origin )
vr = tmgr.vrange()
crs: ccrs.CRS = downscaled.attrs['ccrs']
extent = downscaled.attrs['extent']

fig = plt.figure( figsize=(16,8) )
ax0: Axes = fig.add_subplot( 111, projection=crs )
img0: AxesImage = ax0.imshow( downscaled.data[iband], transform=crs, origin=origin, cmap=cmap, extent=extent, vmin=vr[0], vmax=vr[1] )

plt.show()