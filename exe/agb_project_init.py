import xarray as xa
import holoviews as hv
import panel as pn
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Union, Tuple, Optional, Dict, Callable
pn.extension()
hv.extension('bokeh')

dm: DataManager = DataManager.initialize( "AGB", 'neon' )
tm().reprocess = True
dm.modal.refresh_model = True

block_size = 250
model_dims = 3
nepoch = 1
niter = 100
year= 2015
version = "beta_pmm"
roi = "541567.6_4136443.0_542567.6_4137443.0"

dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/MLBS_{year}_{roi}/MLBS_{year}_Reflectance_reflectance_warp.tif"
tm().block_size = block_size
dm.modal.model_dims = model_dims
dm.modal.modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}"
dm.modal.reduce_nepoch = nepoch
dm.modal.reduce_niter = niter

dm.prepare_inputs()




