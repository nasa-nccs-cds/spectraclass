from .manager import SpatialDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.gui.spatial.aviris.manager import AvirisTileSelector
from spectraclass.gui.spatial.neon.manager import NEONTileSelector
from spectraclass.data.modes import BlockSelectMode
from spectraclass.util.logs import LogManager, lgm, log_timing
from spectraclass.learn.pytorch.progress import ProgressPanel
from panel.widgets import Button, Select
import holoviews as hv
from pathlib import Path
import traitlets as tl
import panel as pn
import os, sys, enum

class AvirisDataManager(SpatialDataManager):
    from spectraclass.gui.spatial.application import Spectraclass
    valid_aviris_bands = tl.List( default_value=[ 0,100000] ).tag(config=True, sync=True)
    version = tl.Unicode( "v2p9" )
    ext = tl.Unicode('.tif').tag(config=True, sync=True)

    MODE = "aviris"
    METAVARS = []
    INPUTS = dict()
    application = Spectraclass

    def __init__(self):
        super(AvirisDataManager, self).__init__()
        self.tile_selector: AvirisTileSelector = None

    def valid_bands(self):
        if self._valid_bands is None:
            self._unpack_valid_bands()
        return self._valid_bands

    def _unpack_valid_bands(self):
        bv0 = 0
        self._valid_bands = []
        for ib, bv1 in enumerate(self.valid_aviris_bands):
            if isinstance(bv1, (list, tuple)): self._valid_bands.append(bv1)
            elif ib % 2 == 1: self._valid_bands.append([bv0, bv1])
            else: bv0 = bv1

    def on_image_change( self, event: Dict ):
        super(AvirisDataManager, self).on_image_change( event )
        self.tile_selector.on_image_change( event )

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
        inames = list(self.image_names.keys())
        assert len( inames ) > 0, f"Error, unable to find any images in the data_dir: {dm().modal.data_dir}"
        base_name = inames[self._active_image]
        if self.ext == ".tif": return f"{base_name}{self.ext}"
        else: return f"ang{base_name}rfl/ang{base_name}_rfl_{self.version}/ang{base_name}_corr_{self.version}{self.ext}"

class NEONDataManager(SpatialDataManager):
    from spectraclass.gui.spatial.application import Spectraclass
    valid_neon_bands = tl.List( default_value=[ 0,100000] ).tag(config=True, sync=True)
    ext = tl.Unicode('.tif').tag(config=True, sync=True)

    MODE = "NEON"
    METAVARS = []
    INPUTS = dict()
    application = Spectraclass

    def __init__(self):
        from spectraclass.reduction.trainer import mt
        super(NEONDataManager, self).__init__()
        self._tile_selector: NEONTileSelector = None
        self._abort = False
        self._progress_panel = ProgressPanel( mt().niter, self.abort_callback)

    @property
    def tile_selector(self):
        if self._tile_selector is None:
            self._tile_selector = NEONTileSelector()
        return self._tile_selector

    def get_tile_selection_gui(self):
        return self.tile_selector.get_tile_selection_gui()

    def abort_callback(self, event ):
        self._abort = True

    def gui(self, **kwargs ):
        return self.tile_selector.gui(**kwargs)

    def get_block_selection(self) -> Optional[Dict]:
        return None if self._tile_selector is None else self._tile_selector.get_block_selection()

    def preprocessing_gui(self):
        from spectraclass.reduction.trainer import mt
        exec_button = pn.widgets.Button( name='Execute',  button_type='success', width=100 )
        exec_button.on_click( self.execute_preprocessing )
        lgm().trace( "#PG: Preprocessing")
        mask_gui = self.gui( mode = BlockSelectMode.LoadMask )
        preprocessing_gui =  pn.WidgetBox( "### Preprocessing", pn.Row(self.parameter_table, exec_button), mt().progress.panel() )
        return pn.Row( mask_gui, preprocessing_gui )

    def execute_preprocessing(self, *args ):
        self.prepare_inputs()

    def valid_bands(self):
        if self._valid_bands is None:
            self._unpack_valid_bands()
        return self._valid_bands

    def _unpack_valid_bands(self):
        bv0 = 0
        self._valid_bands = []
        for ib, bv1 in enumerate(self.valid_neon_bands):
            if isinstance(bv1, (list, tuple)): self._valid_bands.append(bv1)
            elif ib % 2 == 1: self._valid_bands.append([bv0, bv1])
            else: bv0 = bv1

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

