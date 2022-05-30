from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.gui.spatial.map import MapManager, mm
import os
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "img_mgr", 'aviris' )
location = "desktop"
if location == "adapt":
    dm.modal.cache_dir = "/adapt/nobackup/projects/ilab/cache"
    dm.modal.data_dir = "/css/above/daac.ornl.gov/daacdata/above/ABoVE_Airborne_AVIRIS_NG/data/"
elif location == "desktop":
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
else: raise Exception( f"Unknown location: {location}")

dm.modal.ext =  "_img"
dm.proc_type = "cpu"
TileManager.block_size = 256     # 250
ModeDataManager.model_dims = 24  # 16
TileManager.block_index = [0,0]

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

dm.loadCurrentProject()
tm().setBlock( (1,18) )
dm.loadCurrentProject()
