import os
import conda, numbers, time
from typing import List, Union, Tuple, Optional, Dict, Callable
import numpy as np
# conda_file_dir = conda.__file__
# conda_dir = conda_file_dir.split('lib')[0]
# proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
# os.environ["PROJ_LIB"] = proj_lib
import xarray as xa
from matplotlib.image import AxesImage
from cartopy.mpl.geoaxes import GeoAxes
from spectraclass.xext.xgeo import XGeo
import matplotlib.pyplot as plt
import rioxarray as rio
import cartopy.crs as ccrs
t0 = time.time()

iband = 200
origin = "upper"
cmap="jet"
epsg = 4326
reproject_using = "cartopy"

gcrs = ccrs.PlateCarree()
wcrs = ccrs.epsg(3857)
SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
utm_data_array: xa.DataArray = rio.open_rasterio( SpectralDataFile, chunks=True ) # TileManager.read_data_layer( SpectralDataFile, origin, nodata_fill=0 )
utm_band_array = utm_data_array[iband].squeeze()

band_array: np.ndarray = utm_band_array.data
band_extent: List[int] = utm_data_array.xgeo.bounds()

ax0: GeoAxes = plt.axes( projection=wcrs )
img0: AxesImage = ax0.imshow( band_array, transform=wcrs, origin=origin, cmap=cmap, extent=band_extent )
ax0.set_extent( band_extent, wcrs )
plt.show()