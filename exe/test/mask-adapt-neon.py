from spectraclass.data.base import DataManager
from spectraclass.learn.pytorch.trainer import mpt
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.modes import BlockSelectMode
from spectraclass.model.labels import LabelsManager, lm

tm().reprocess = True
block_size = 100
year= 2016
version = "beta_pmm"
roi = "364203.7_4305235.0_365203.7_4306235.0"
selection_mode: BlockSelectMode = BlockSelectMode.CreateMask

dm: DataManager = DataManager.initialize( "AGB1", 'neon', selection_mode )
dm.analysis_data_source = "spectral"
dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/SERC_{year}_{roi}/SERC_{year}_Reflectance_reflectance_warp.tif"
tm().block_size = block_size
mpt().niter = 50
mpt().modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}"

classes = { 1: ('forest', "green"),  0: ('non-forest', "magenta") }
lm().setLabels( classes )

dm.preprocess_gui()