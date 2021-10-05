import os
import conda, numbers
import numpy as np

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio.crs as rcrs
import geopandas as gpd

iband = 150
desis_image_index = 4
epsg = 4326
origin = "upper"
cmap="jet"

rgcrs: rcrs.CRS = rcrs.CRS.from_epsg( 4326 )

SpectralDataFile = f"/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_00{desis_image_index}-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
boundaries_shp = "/Users/tpmaxwel/Development/Data/gis/Maryland/Maryland_Physical_Boundaries_-_County_Boundaries_(Generalized).shp"

shapefile: gpd.GeoDataFrame = gpd.read_file(boundaries_shp)
shape_crs = shapefile.crs
map_dframe: gpd.GeoDataFrame = shapefile.to_crs( epsg=epsg )
map_crs = map_dframe.crs
utm_data_array: xa.DataArray = rio.open_rasterio( SpectralDataFile, mask_and_scale=True )  # TileManager.read_data_layer( SpectralDataFile, origin, nodata_fill=0 )
utm_band_data: xa.DataArray = utm_data_array[iband].squeeze()
data_array: xa.DataArray = utm_band_data.rio.reproject( rgcrs, shape=utm_band_data.shape )
# tcrs: ccrs.CRS = data_array.attrs['ccrs']
# extent = data_array.attrs['extent']

ax0: Axes = plt.axes()
image: QuadMesh = data_array.plot( ax=ax0, alpha=0.7, cmap="jet" )
map: AxesImage = map_dframe.plot( ax=ax0, alpha=0.3, color="green"  )
plt.show()

#band_data: np.ndarray = data_array.data
#img0: AxesImage = ax0.imshow( band_data, cmap=cmap )
# ax0.add_geometries( boundaries_geo.geometries(), crs=boundaries_crs, alpha=0.5, facecolor="green" )
# ax0.gridlines( crs=gcrs, draw_labels=True, alpha=0.5 )
#print(f"Plotting Image, shape = {data_array.shape[1:]}")
