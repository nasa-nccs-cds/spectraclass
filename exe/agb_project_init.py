from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
dm: DataManager = DataManager.initialize( "AGB", 'neon' )

block_size = 250
model_dims = 12
nepochs = 100
year= 2015
version = "beta_pmm"
roi = "541567.6_4136443.0_542567.6_4137443.0"

dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/MLBS_{year}_{roi}/MLBS_{year}_Reflectance_reflectance_warp.tif"
TileManager.block_size = block_size
TileManager.reprocess = False
dm.modal.refresh_model = False
dm.modal.model_dims = model_dims
dm.modal.modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}"
dm.modal.reduce_nepoch = nepochs

dm.prepare_inputs()




