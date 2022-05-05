from .manager import SpatialDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
from pathlib import Path
import traitlets as tl
import os, sys

class AvirisDataManager(SpatialDataManager):
    from spectraclass.gui.spatial.application import Spectraclass
    valid_aviris_bands = tl.List( default_value=[ 0,100000] ).tag(config=True, sync=True)
    version = tl.Unicode( "v2p9" )

    MODE = "aviris"
    METAVARS = []
    INPUTS = dict()
    application = Spectraclass

    def __init__(self):
        super(AvirisDataManager, self).__init__()
        self.VALID_BANDS = []
        self._unpack_valid_bands()

    def _unpack_valid_bands(self):
        bv0 = 0
        for ib, bv1 in enumerate(self.valid_aviris_bands):
            if ib % 2 == 1: self.VALID_BANDS.append( [ bv0, bv1 ] )
            bv0 = bv1

    @classmethod
    def extract_image_name( cls, image_path: str ) -> str:
        basename = Path(image_path).stem
        if basename.startswith("ang"): basename = basename[3:]
        return basename.split("_")[0]

    @property
    def default_images_glob(self):
        return f"ang*rfl/ang*_rfl_{self.version}/ang*_corr_{self.version}{self.ext}"

    @property
    def image_name(self):
        base_name = self.image_names[self._active_image]
        return f"ang{base_name}rfl/ang{base_name}_rfl_{self.version}/ang{base_name}_corr_{self.version}{self.ext}"

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
    MODE = "desis"
    METAVARS = []
    INPUTS = dict()
    VALID_BANDS = [ [5, sys.maxsize], ]
    application = Spectraclass

    def __init__(self):
        super(DesisDataManager, self).__init__()

    @property
    def default_images_glob(self):
        return "*-SPECTRAL_IMAGE" + self.ext

