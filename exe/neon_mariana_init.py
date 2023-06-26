from spectraclass.data.base import DataManager
from spectraclass.reduction.vae.trainer import mt
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Union, Tuple, Optional, Dict, Callable

dm: DataManager = DataManager.initialize( "AGB", 'neon' )
tm().reprocess = False
mt().refresh_model = True

block_size = 100
model_dims = 3
reduction_factor = 3

niter = 30
nepoch = 5
learning_rate = 0.01
activation = "relu"

focus_nepoch = 0    # 5
focus_ratio = 10
focus_threshold = 0.1

anomaly = "diff"
year = 2016

version = "beta_pmm"
roi = "364203.7_4305235.0_365203.7_4306235.0"

# /explore/nobackup/projects/ilab/data/AGB/test/beta_pmm/SERC_2016_364203.7_4305235.0_365203.7_4306235.0/SERC_2016_Reflectance_reflectance_warp.tif

dm.proc_type = "cpu"
dm.modal.anomaly = anomaly
dm.modal.images_glob = f"AGB/test/{version}/SERC_{year}_{roi}/SERC_{year}_Reflectance_reflectance_warp.tif"
tm().block_size = block_size
mt().model_dims = model_dims
mt().modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}.{anomaly}.{reduction_factor}"
mt().nepoch = nepoch
mt().focus_nepoch = focus_nepoch
mt().focus_ratio = focus_ratio
mt().focus_threshold = focus_threshold
mt().activation = activation
mt().reduction_factor = reduction_factor
mt().niter = niter


dm.prepare_inputs()




