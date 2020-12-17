from .manager import SpatialDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable

class AvirisDataManager(SpatialDataManager):
    MODE = "aviris"
    METAVARS = []
    INPUTS = dict( embedding='scaled_specs', directory = [  "target_names", "obsids" ], plot= dict( y="specs", x='spectra_x_axis' ) )


    def __init__(self):
        super(AvirisDataManager, self).__init__()

class DesisDataManager(SpatialDataManager):
    MODE = "desis"
    METAVARS = []
    INPUTS = dict( embedding='scaled_specs', directory = [  "target_names", "obsids" ], plot= dict( y="specs", x='spectra_x_axis' ) )

    def __init__(self):
        super(DesisDataManager, self).__init__()
