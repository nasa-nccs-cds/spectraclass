from .manager import SpatialDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.gui.spatial.aviris.manager import AvirisTileSelector
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
        self.tile_selector: AvirisTileSelector = None

    def _unpack_valid_bands(self):
        bv0 = 0
        for ib, bv1 in enumerate(self.valid_aviris_bands):
            if isinstance(bv1, (list, tuple)): self.VALID_BANDS.append( bv1 )
            elif ib % 2 == 1: self.VALID_BANDS.append( [ bv0, bv1 ] )
            else: bv0 = bv1

    def gui(self):
        if self.ext == "_img":
            if self.tile_selector is None:
                self.tile_selector = AvirisTileSelector()
            return self.tile_selector.gui()
        else:
            return SpatialDataManager.gui( self )

    @classmethod
    def extract_image_name( cls, image_path: str ) -> str:
        basename = Path(image_path).stem
        if basename.startswith("ang"):
            basename = basename[3:]
        if not image_path.endswith(".tif"):
            basename =  basename.split("_")[0]
        return basename

    @property
    def default_images_glob(self):
        return f"ang*rfl/ang*_rfl_{self.version}/ang*_corr_{self.version}{self.ext}"

    @property
    def image_name(self):
        from spectraclass.data.base import DataManager, dm
        assert len( self.image_names ) > 0, f"Error, unable to find any images in the data_dir: {dm().modal.data_dir}"
        base_name = self.image_names[self._active_image]
        if self.ext == ".tif": return f"{base_name}{self.ext}"
        else: return f"ang{base_name}rfl/ang{base_name}_rfl_{self.version}/ang{base_name}_corr_{self.version}{self.ext}"

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
    def extension(self):
        return "-SPECTRAL_IMAGE" + self.ext

