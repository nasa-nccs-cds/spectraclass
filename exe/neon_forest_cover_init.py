from spectraclass.data.base import DataManager
from spectraclass.reduction.vae.trainer import mt
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "AGB", 'neon' )
tm().reprocess = True
mt().refresh_model = True

block_size = 150
model_dims = 3
nepoch = 10
focus_nepoch = 5
focus_ratio = 10
focus_threshold = 0.1
niter = 10
year= 2015
version = "beta_pmm"
roi = "541567.6_4136443.0_542567.6_4137443.0"

dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/MLBS_{year}_{roi}/MLBS_{year}_Reflectance_reflectance_warp.tif"
tm().block_size = block_size
mt().model_dims = model_dims
mt().modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}"
mt().nepoch = nepoch
mt().focus_nepoch = focus_nepoch
mt().focus_ratio = focus_ratio
mt().focus_threshold = focus_threshold
mt().niter = niter


dm.prepare_inputs()




