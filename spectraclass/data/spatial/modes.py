from .manager import SpatialDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable

class AvirisDataManager(SpatialDataManager):
    MODE = "aviris"
    METAVARS = []
    INPUTS = dict()
    VALID_BANDS = [[3, 193], [214, 283], [319, 421]]

    def __init__(self):
        super(AvirisDataManager, self).__init__()

class DesisDataManager(SpatialDataManager):
    MODE = "desis"
    METAVARS = []
    INPUTS = dict()

    def __init__(self):
        super(DesisDataManager, self).__init__()
