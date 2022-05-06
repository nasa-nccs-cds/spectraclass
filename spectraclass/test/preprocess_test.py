import os
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.data.base import ModeDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
host = "desktop" # "desktop" # "laptop"

dm: DataManager = DataManager.initialize( "demo2", 'aviris' )
if (host == "laptop"):
    dm.modal.data_dir = os.path.expanduser("~/Development/Data/Aviris/processed")
    dm.modal.cache_dir = os.path.expanduser("~/Development/Cache")
else:
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Volumes/Shared/Data/Aviris"

TileManager.block_size = 250
ModeDataManager.model_dims = 16
TileManager.reprocess = True

dm.generate_metadata()