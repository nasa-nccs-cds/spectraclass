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
    MODE = "desis"
    METAVARS = []
    INPUTS = dict()
    VALID_BANDS = [ [5, sys.maxsize], ]

    def __init__(self):
        super(DesisDataManager, self).__init__()

    def getFilePath(self, use_tile: bool ) -> str:
        base_dir = self.tiles.data_dir
        base_file = self.tiles.tileName() if use_tile else self.tiles.image_name
        base_image = f"{base_dir}/{base_file}-SPECTRAL_IMAGE"
        mdata_file = f"{base_dir}/{base_file}-METADATA.xml"
        if not os.path.isfile( base_image + ".xml" ) and os.path.isfile( mdata_file ):
            os.system(f'ln -s "{mdata_file}" "{base_image}.xml"')
        return f"{base_image}.tif"
