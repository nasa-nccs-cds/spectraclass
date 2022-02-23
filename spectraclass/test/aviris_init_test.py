from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.data.spatial.tile.manager import TileManager

dm: DataManager = DataManager.initialize( "demo2", 'aviris' )
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Volumes/Shared/Data/Aviris"
dm.modal.image_names = ["ang20170720t004130_corr_v2p9"]
dm.proc_type = "cpu"
TileManager.block_size = 256
ModeDataManager.model_dims = 24
TileManager.block_index = [1,1]

dm.loadCurrentProject()
