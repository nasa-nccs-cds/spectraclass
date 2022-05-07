import os, time
import xarray as xa
import rioxarray as rio
from spectraclass.data.spatial.tile.manager import TileManager, tm

iband = 0
cdim = "band"
SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
result_dir = "/Users/tpmaxwel/Development/Data/raster/tiles"

t0 = time.time()
base_name = os.path.basename( os.path.splitext(SpectralDataFile)[0] )
result_path = os.path.join( result_dir, f"{base_name}.nc" )
os.makedirs( result_dir, exist_ok=True )
raster: xa.DataArray = rio.open_rasterio( SpectralDataFile, default_name='z' )
[xc,yc] = [ raster.coords[raster.dims[ic]].values for ic in (2,1) ]
print( f" sref = {raster.spatial_ref.crs_wkt}")
#print( f" XC[80:85] = {[xc[]:.4f}")
#print( f" YC[80:85] = {yc[80:90]:.4f}")
raster = raster.rio.reproject( TileManager.crs )
# raster = raster.chunk( chunks={ cdim: 1 } )
raster.to_netcdf( result_path )

print( f"Completed generating file '{result_path}' in total time = {time.time()-t0} sec.")


# raster: xa.DataArray = TileManager.to_standard_form( data, origin, ccrs=False, name='z' )
# raster = raster.chunk( { raster.dims[0]: 1 } )


