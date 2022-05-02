import time
import xarray as xa
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import rioxarray as rio
t0 = time.time()

iband = 0
origin = "upper"
cmap="jet"
SpectralDataFile = "/Users/tpmaxwel/Development/Data/Aviris/ang20170720t004130rfl/ang20170720t004130_rfl_v2p9/ang20170720t004130_corr_v2p9.tif"

data_array: xa.DataArray = rio.open_rasterio( SpectralDataFile ) # , chunks=True ) # TileManager.read_data_layer( SpectralDataFile, origin, nodata_fill=0 )
band_array = data_array[iband].squeeze()

ax0 = plt.axes( ) # projection=ax_crs )
img0: AxesImage = ax0.imshow( band_array, origin=origin, cmap=cmap )
print(f"Plotting Image, shape = {data_array.shape[1:]}, time = {time.time()-t0} sec. " )
plt.show()