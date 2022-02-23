from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.application.controller import app, SpectraclassController
from spectraclass.model.labels import LabelsManager, lm
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "demo4", 'tess' )
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Volumes/Shared/Data/tess"
dm.proc_type = "cpu"
ModeDataManager.model_dims = 24

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

dm.loadCurrentProject()