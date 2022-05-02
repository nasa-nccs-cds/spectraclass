import time
import numpy as np
import xarray as xa
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import rioxarray as rio
t0 = time.time()

iband = 0
origin = "upper"
cmap="jet"
SpectralDataFile = "/Users/tpmaxwel/Development/Data/Aviris/ang20170720t004130rfl/ang20170720t004130_rfl_v2p9/ang20170720t004130_corr_v2p9.tif"

data_array: xa.DataArray = rio.open_rasterio( SpectralDataFile, chunks=True )
band_array: np.ndarray = data_array[iband].values.squeeze()
nodata = data_array.attrs.get('_FillValue')
band_array[band_array == nodata] = np.nan
valid_mask = ~np.isnan(band_array)
nvalid = np.count_nonzero(valid_mask)
ntotal = valid_mask.size
print(f"Read data, shape = {band_array.shape}, #valid = {nvalid}/{ntotal}, fraction = {nvalid/ntotal}, read time = {(time.time()-t0)/60} min " )
print( "Spatial Ref: ")
print( data_array.spatial_ref )
print( "Coords: ")
for k,c in data_array.coords.items():
    print( f"{k}: {c.shape}" )

ax0 = plt.axes( )
img0: AxesImage = ax0.imshow( band_array, origin=origin, cmap=cmap )
plt.show()