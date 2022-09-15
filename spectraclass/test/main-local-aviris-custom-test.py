from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.learn.cluster.manager import ClusterManager, clm
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize("img_mgr", 'aviris')
location = "desktop"
if location == "adapt":
    dm.modal.cache_dir = "/adapt/nobackup/projects/ilab/cache"
    dm.modal.data_dir = "/css/above/daac.ornl.gov/daacdata/above/ABoVE_Airborne_AVIRIS_NG/data/"
elif location == "desktop":
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
else:
    raise Exception(f"Unknown location: {location}")

block_size = 150
method = "aec"  # "vae"
model_dims = 32
reprocess_data = False

dm: DataManager = DataManager.initialize("img_mgr", 'aviris')
TileManager.block_size = block_size
dm.modal.model_dims = model_dims
dm.modal.reduce_anom_focus = 0.20
dm.modal.reduce_nepoch = 3
dm.modal.reduce_focus_nepoch = 10
dm.modal.reduce_niter = 1
dm.modal.refresh_model = False
dm.modal.modelkey = f"b{block_size}.{method}"
dm.modal.reduce_focus_ratio = 10.0
dm.modal.reduce_dropout = 0.0
dm.modal.reduce_learning_rate = 1e-4
ClusterManager.modelid = "kmeans"

dm.loadCurrentProject()
dm.modal.initialize_dimension_reduction( refresh=reprocess_data )
cluster_image: xa.DataArray = clm().run_cluster_model( dm.getModelData() )
clm().rescale( 0, 0.5 )






