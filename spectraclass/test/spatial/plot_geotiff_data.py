import os
import conda, numbers, time
from typing import List, Union, Tuple, Optional, Dict, Callable
import numpy as np

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
from matplotlib.image import AxesImage
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio.crs as rcrs
from cartopy.io.shapereader import Reader as ShapeReader
from spectraclass.gui.spatial.widgets.tiles import TileSelector, TileManager
import cartopy.crs as ccrs
t0 = time.time()

iband = 200
origin = "upper"
cmap="jet"
epsg = 4326
reproject_using = "cartopy"

gcrs = ccrs.PlateCarree()
rgcrs: rcrs.CRS = rcrs.CRS.from_dict( dict(gcrs.proj4_params) )

SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
boundaries_shp = "/Users/tpmaxwel/Development/Data/gis/Maryland/Maryland_Physical_Boundaries_-_County_Boundaries_(Generalized).shp"
boundaries_geo = ShapeReader( boundaries_shp )
recs = boundaries_geo.records()
attrs = next(recs).attributes
boundaries_crs: ccrs.CRS = TileManager.get_shp_crs( boundaries_shp )

utm_data_array: xa.DataArray = rio.open_rasterio( SpectralDataFile, chunks=True ) # TileManager.read_data_layer( SpectralDataFile, origin, nodata_fill=0 )
utm_band_array = utm_data_array[iband].squeeze()

if reproject_using == "rio":
    band_array = utm_band_array.rio.reproject( gcrs.proj4_init, shape=utm_band_array.shape )
    band_crs = rgcrs
    band_extent = []
else:
    band_array: np.ndarray = utm_band_array.data
    band_crs: ccrs.CRS = utm_data_array.attrs['ccrs']
    band_extent: List[int] = utm_data_array.attrs['extent']

ax_crs = gcrs if reproject_using == "cartopy" else band_crs
ax0: GeoAxes = plt.axes( projection=ax_crs )
img0: AxesImage = ax0.imshow( band_array, transform=band_crs, origin=origin, cmap=cmap, extent=band_extent )
ax0.set_extent( band_extent, band_crs )
# ax0.add_geometries( boundaries_geo.geometries(), crs=boundaries_crs, alpha=0.5, facecolor="green" )
ax0.gridlines( crs=gcrs, draw_labels=True, alpha=0.5 )
print(f"Plotting Image, shape = {utm_data_array.shape[1:]}, time = {time.time()-t0} sec. " )
plt.show()