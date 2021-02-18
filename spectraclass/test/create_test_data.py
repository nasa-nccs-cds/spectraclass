import random, os, numpy as np
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
import time, xarray as xa

dataset_type = "chr"
dir_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs( os.path.join( dir_path, "data", dataset_type ), exist_ok=True )

dm: DataManager = DataManager.initialize("demo1",'keelin')
project_data: xa.Dataset = dm.loadCurrentProject("main")
bands: xa.DataArray = tm().getBlock().data
bands.attrs['dsid'] = bands.name
bands.name = "data"
model, x, y =project_data.model, bands.x,  bands.y
reduced_samples: xa.DataArray = project_data.reduction.transpose()
reduced: np.ndarray = reduced_samples.data.reshape( model.size, y.size, x.size )
xReduced: xa.DataArray = xa.DataArray( name="data", data=reduced, dims=['model','y','x'], coords = dict( model=model, x=x, y=y ) )

for iB in range( bands.shape[0] ):
    path = os.path.join( dir_path, "data", dataset_type, f"ks_raw_{iB}.nc4")
    print( f"Writing file: {path}")
    slice = bands[iB]
    slice.attrs = bands.attrs
    slice.to_netcdf( path, format="NETCDF4" )

for iB in range( xReduced.shape[0] ):
    path = os.path.join( dir_path, "data", dataset_type, f"ks_pca_{iB}.nc4")
    print(f"Writing file: {path}")
    slice = xReduced[iB]
    slice.attrs = bands.attrs
    slice.to_netcdf( path, format="NETCDF4" )