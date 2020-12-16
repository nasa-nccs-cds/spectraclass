from .manager import SpatialDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable

class AvirisDataManager(SpatialDataManager):
    MODE = "aviris"

    def __init__(self):
        super(AvirisDataManager, self).__init__()

class DesisDataManager(SpatialDataManager):
    MODE = "desis"

    def __init__(self):
        super(DesisDataManager, self).__init__()
