import xarray as xa
import holoviews as hv
import panel as pn
from spectraclass.model.labels import lm
from spectraclass.gui.spatial.viewer import hvSpectraclassGui, sgui
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.reduction.trainer import mt
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.modes import BlockSelectMode

hv.extension('bokeh')

dm: DataManager = DataManager.initialize( "AGB", 'neon' )
tm().reprocess = False
mt().refresh_model = False

block_size = 150
model_dims = 3
nepoch = 1
niter = 100
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

dm.loadCurrentProject()
classes = [ ('air', "cyan"),
            ('water', "blue"),
            ('fire', "magenta"),
            ('earth', "green") ]

lm().setLabels( classes )

viewer: hvSpectraclassGui   = sgui().init()
viewer.panel(mode=BlockSelectMode.LoadTile)