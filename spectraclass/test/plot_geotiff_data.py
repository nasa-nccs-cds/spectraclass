import os
import conda, numbers
import numpy as np

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
from matplotlib.image import AxesImage
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
from spectraclass.gui.spatial.widgets.tiles import TileSelector, TileManager
import cartopy.crs as ccrs

iband = 200
origin = "upper"
cmap="jet"
SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"

data_array: xa.DataArray = TileManager.read_data_layer( SpectralDataFile, origin, nodata_fill=0 )
crs: ccrs.CRS = data_array.attrs['ccrs']
extent = data_array.attrs['extent']

fig = plt.figure( figsize=(16,8) )
ax0: GeoAxes = fig.add_subplot( 111, projection=crs )
raw_band_data: np.ndarray = data_array.data[iband]
band_data: np.ndarray = raw_band_data/raw_band_data.max()
img0: AxesImage = ax0.imshow( band_data, transform=crs, origin=origin, cmap=cmap, extent=extent )
ax0.gridlines()
print(f"Plotting Image, shape = {data_array.shape[1:]}")
plt.show()