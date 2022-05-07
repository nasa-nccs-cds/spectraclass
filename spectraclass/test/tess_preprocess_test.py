from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.data.unstructured.modes import TessDataManager
from spectraclass.data.spatial.tile.manager import TileManager
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "demo4", 'tess' )
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Volumes/Shared/Data"
dm.modal.dset_name = "s20"
dm.proc_type = "cpu"
ModeDataManager.model_dims = 24
dm.prepare_inputs( reprocess=True )

