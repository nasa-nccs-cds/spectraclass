from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.gui.spatial.map import MapManager, mm
import os
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "demo2", 'desis' )
location = "desktop"
if (location == "laptop"):
    dm.modal.data_dir = os.path.expanduser("~/Development/Data/DESIS")
    dm.modal.cache_dir = os.path.expanduser("/Volumes/archive/Cache")
elif (location == "desktop"):
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Volumes/Shared/Data/DESIS"
else:
    raise Exception( f"Unknown location: {location}")

dm.proc_type = "cpu"
TileManager.block_size = 256     # 250
ModeDataManager.model_dims = 24  # 16
TileManager.block_index = [1,1]

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

dm.loadCurrentProject()

