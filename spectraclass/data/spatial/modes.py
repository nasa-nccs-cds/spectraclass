from .manager import SpatialDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
import os, sys

class AvirisDataManager(SpatialDataManager):
    from spectraclass.gui.spatial.application import Spectraclass

    MODE = "aviris"
    METAVARS = []
    INPUTS = dict()
    VALID_BANDS = [ [5, 193], [214, 283], [319, sys.maxsize] ]
    application = Spectraclass

    def __init__(self):
        super(AvirisDataManager, self).__init__()

class KeelinDataManager(SpatialDataManager):
    from spectraclass.gui.spatial.application import Spectraclass

    MODE = "keelin"
    METAVARS = []
    INPUTS = dict()
    application = Spectraclass

    def __init__(self):
        super(KeelinDataManager, self).__init__()

class DesisDataManager(SpatialDataManager):
    from spectraclass.gui.spatial.application import Spectraclass
    MODE = "DESIS"
    METAVARS = []
    INPUTS = dict()
    VALID_BANDS = [ [5, sys.maxsize], ]
    application = Spectraclass

    def __init__(self):
        super(DesisDataManager, self).__init__()

    def getFilePath(self, use_tile: bool ) -> str:
        if use_tile:
            mode_dir = f"{self.cache_dir}/{self.MODE}"
            os.makedirs( mode_dir, 0o777, exist_ok=True )
            return f"{mode_dir}/{self.tiles.getTileFileName()}"
        else:
            return f"{self.data_dir}/{self.tiles.image_name}-SPECTRAL_IMAGE.tif"
