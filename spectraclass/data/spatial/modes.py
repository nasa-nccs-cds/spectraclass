from .manager import SpatialDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
import os, sys

class AvirisDataManager(SpatialDataManager):
    MODE = "aviris"
    METAVARS = []
    INPUTS = dict()
    VALID_BANDS = [ [5, 193], [214, 283], [319, sys.maxsize] ]

    def __init__(self):
        super(AvirisDataManager, self).__init__()

class DesisDataManager(SpatialDataManager):
    MODE = "DESIS"
    METAVARS = []
    INPUTS = dict()
    VALID_BANDS = [ [5, sys.maxsize], ]

    def __init__(self):
        super(DesisDataManager, self).__init__()

    def getFilePath(self, use_tile: bool ) -> str:
        data_dir = os.path.join( self.data_dir, self.MODE )
        base_file = self.tiles.getTileFileName() if use_tile else self.tiles.image_name
        return f"{data_dir}/processed/{base_file}"
