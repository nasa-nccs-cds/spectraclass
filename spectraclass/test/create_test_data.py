import random, os, numpy as np
from typing import List, Optional, Dict, Type
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
import time, xarray as xa

save_in_project = False
dm: DataManager = DataManager.initialize("demo1",'keelin')
dataset_type = "chr"
dir_path = os.path.dirname(os.path.realpath(__file__)) if save_in_project else os.path.join( dm.cache_dir, "test" )
os.makedirs( os.path.join( dir_path, "data", dataset_type ), exist_ok=True )

project_data: xa.Dataset = dm.loadCurrentProject("main")
bands: xa.DataArray = tm().getBlock().data
bIdx: List[int] = tm().block_index
bands.attrs['dsid'] = bands.name
bands.name = "data"
model, x, y =project_data.model, bands.x,  bands.y
reduced_samples: xa.DataArray = project_data.reduction.transpose()
reduced: np.ndarray = reduced_samples.data.reshape( model.size, y.size, x.size )
xReduced: xa.DataArray = xa.DataArray( name="data", data=reduced, dims=['model','y','x'], coords = dict( model=model, x=x, y=y ) )

for iB in range( bands.shape[0] ):
    path = os.path.join( dir_path, "data", dataset_type, f"ks_raw_{bIdx[0]}_{bIdx[1]}_{iB}.nc4")
    print( f"Writing file: {path}")
    slice = bands[iB]
    slice.attrs = bands.attrs
    slice.to_netcdf( path, format="NETCDF4" )

for iB in range( xReduced.shape[0] ):
    path = os.path.join( dir_path, "data", dataset_type, f"ks_pca_{bIdx[0]}_{bIdx[1]}_{iB}.nc4")
    print(f"Writing file: {path}")
    slice = xReduced[iB]
    slice.attrs = bands.attrs
    slice.to_netcdf( path, format="NETCDF4" )