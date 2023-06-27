from spectraclass.data.base import DataManager
from spectraclass.reduction.trainer import mt
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "AGB1", 'neon' )
tm().reprocess = True
mt().refresh_model = True

block_size = 100
model_dims = 3
nepoch = 10
focus_nepoch = 5
focus_ratio = 10
focus_threshold = 0.1
niter = 100
year= 2016
version = "beta_pmm"
roi = "364203.7_4305235.0_365203.7_4306235.0"

# /explore/nobackup/projects/ilab/data/AGB/test/beta_pmm/SERC_2016_364203.7_4305235.0_365203.7_4306235.0/SERC_2016_Reflectance_reflectance_warp.tif

dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/SERC_{year}_{roi}/SERC_{year}_Reflectance_reflectance_warp.tif"
tm().block_size = block_size
mt().model_dims = model_dims
mt().modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}"
mt().nepoch = nepoch
mt().focus_nepoch = focus_nepoch
mt().focus_ratio = focus_ratio
mt().focus_threshold = focus_threshold
mt().niter = niter
mt().log_step = 50


dm.prepare_inputs()




