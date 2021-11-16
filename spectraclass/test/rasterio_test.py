import contextily as cx
from matplotlib.pyplot import figure, draw, pause
from xyzservices import TileProvider
from matplotlib.axes import Axes
import xarray as xa
import os, logging
import rioxarray as rio
import matplotlib.pyplot as plt

log_file = os.path.expanduser('~/.spectraclass/logging/geospatial.log')
file_handler = logging.FileHandler(filename=log_file, mode='w')
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

result_dir = "/Users/tpmaxwel/Development/Data/raster/tiles"
SpectralDataset = f"{result_dir}/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.nc"
raster: xa.DataArray = rio.open_rasterio( SpectralDataset )
classes = dict( water='blue', vegetation='green', urban='grey' )
band = 100


fig, ax = plt.subplots(1,1, figsize=(8.0,8.0) )
raster[band].plot( ax=ax )
cx.add_basemap( ax, source=cx.providers.Esri.WorldImagery )
plt.show()

