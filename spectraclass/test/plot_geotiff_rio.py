import os
import conda, numbers, time
from typing import List, Union, Tuple, Optional, Dict, Callable
import numpy as np

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join( conda_dir, 'share', 'proj' )
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio.crs as rcrs
t0 = time.time()

iband = 200
origin = "upper"
cmap="jet"
epsg = 4326

SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
boundaries_shp = "/Users/tpmaxwel/Development/Data/gis/Maryland/Maryland_Physical_Boundaries_-_County_Boundaries_(Generalized).shp"

utm_data_array: xa.DataArray = rio.open_rasterio( SpectralDataFile, chunks=True ) # TileManager.read_data_layer( SpectralDataFile, origin, nodata_fill=0 )
raster: rio.raster_array.RasterArray = utm_data_array[iband].squeeze()

# band_array = utm_band_array.rio.reproject( gcrs.proj4_init, shape=utm_band_array.shape )
# band_crs = rgcrs
# band_extent = []

ax0 = plt.axes()
raster.plot( ax = ax0 )
# img0: AxesImage = ax0.imshow( utm_band_array, cmap=cmap )
# ax0.set_extent( band_extent, band_crs )
# # ax0.add_geometries( boundaries_geo.geometries(), crs=boundaries_crs, alpha=0.5, facecolor="green" )
# ax0.gridlines( crs=gcrs, draw_labels=True, alpha=0.5 )
print(f"Plotting Image, shape = {utm_data_array.shape[1:]}, time = {time.time()-t0} sec. " )
plt.show()