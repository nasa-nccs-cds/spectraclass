from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.application.controller import SpectraclassController, app
import os
host = "desktop"

dm: DataManager = DataManager.initialize( "demo2", 'aviris' )
if (host == "laptop"):
    dm.modal.data_dir = os.path.expanduser("~/Development/Data/Aviris/processed")
    dm.modal.cache_dir = os.path.expanduser("/Volumes/archive/Cache")
else:
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Volumes/Shared/Data/Aviris"

dm.modal.image_names = ["20170720t004130"]
dm.proc_type = "cpu"
TileManager.block_size = 256  # 250
ModeDataManager.model_dims = 24  # 16
TileManager.block_index = [0, 2]

dm.loadCurrentProject()

controller: SpectraclassController = app()
controller.gui()
