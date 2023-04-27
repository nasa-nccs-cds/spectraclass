from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
import sys

dm: DataManager = DataManager.initialize( "AGB", 'neon' )

dm.modal.cache_dir = "/explore/nobackup/projects/ilab/cache"
dm.modal.data_dir  = "/explore/nobackup/projects/ilab/data"

#dm.modal.cache_dir = "/Volumes/archive/spectraclass/logs"
#dm.modal.data_dir  = "/Volumes/archive/data/"

block_size = 150
method = "aec" # "vae"
model_dims = 32
year= 2015
version = "beta_pmm"
roi = "541567.6_4136443.0_542567.6_4137443.0"

dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/MLBS_{year}_{roi}/MLBS_{year}_Reflectance_reflectance_warp.tif"
# dm.modal.images_glob = f"AGB/test/{version}/MLBS_{year}_Reflectance_reflectance_warp.tif"
TileManager.block_size = block_size
TileManager.reprocess = True
dm.modal.model_dims = model_dims
dm.modal.reduce_method = method
dm.modal.reduce_nepoch = 2
dm.modal.reduce_focus_nepoch = 0
dm.modal.reduce_niter = 12
dm.modal.reduce_focus_ratio = 10.0
dm.modal.reduce_dropout = 0.0
dm.modal.reduce_learning_rate = 1e-4
dm.modal.refresh_model = True
dm.modal.reduce_nblocks = 1000
dm.modal.reduce_nimages = 100
dm.modal.modelkey = f"b{block_size}.{version}.{year}.{roi}"

dm.prepare_inputs()




