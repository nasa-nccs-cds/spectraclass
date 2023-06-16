from spectraclass.data.base import DataManager
from spectraclass.reduction.trainer import mt
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "AGB", 'neon' )
tm().reprocess = True
mt().refresh_model = True

block_size = 150
model_dims = 3
nepoch = 1
niter = 100
init_wts = 1.0
init_bias = 1.0
year= 2015
version = "beta_pmm"
roi = "541567.6_4136443.0_542567.6_4137443.0"

dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/MLBS_{year}_{roi}/MLBS_{year}_Reflectance_reflectance_warp.tif"
tm().block_size = block_size
mt().model_dims = model_dims
mt().modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}"
mt().nepoch = nepoch
mt().niter = niter
mt().init_wts_mag = init_wts
mt().init_bias_mag = init_bias

dm.prepare_inputs()




