from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.data.spatial.tile.manager import tm
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "img_mgr", 'aviris' )
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"

block_size = 200
method = "aec" # "vae"
model_dims = 32
version = "v2p9"
month = "201707"

dm.proc_type = "skl"
dm.modal.images_glob = f"ang{month}*rfl/ang*_rfl_{version}/ang*_corr_{version}_img"
TileManager.block_size = block_size
TileManager.block_index = [0,5]
AvirisDataManager.version = version
dm.modal.valid_aviris_bands = [ [5,193], [214,283], [319,10000] ]
dm.modal.model_dims = model_dims
dm.modal.reduce_method = method
dm.modal.reduce_nepoch = 3
dm.modal.reduce_focus_nepoch = 10
dm.modal.reduce_niter = 1
dm.modal.reduce_focus_ratio = 10.0
dm.modal.reduce_dropout = 0.0
dm.modal.reduce_learning_rate = 1e-4
dm.modal.refresh_model = False
dm.modal.modelkey = f"b{block_size}.{method}"

dm.loadCurrentProject()
classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

lm().setLabels( classes )
dm.modal.initialize_dimension_reduction()

block: Block = tm().getBlock()
points, grid = block.getPointData()
block_data: xa.DataArray = block.data
pid_array = block.get_gid_array()
pass