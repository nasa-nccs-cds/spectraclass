from spectraclass.data.base import DataManager
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.gui.spatial.map import MapManager, mm
import numpy as np
from spectraclass.data.spatial.tile.manager import TileManager, tm
import xarray as xa

image_index = 1
dm: DataManager = DataManager.initialize("img_mgr", 'aviris')

location = "desktop"
if location == "adapt":
    dm.modal.cache_dir = "/adapt/nobackup/projects/ilab/cache"
    dm.modal.data_dir = "/css/above/daac.ornl.gov/daacdata/above/ABoVE_Airborne_AVIRIS_NG/data/"
elif location == "desktop":
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
else:
    raise Exception(f"Unknown location: {location}")

block_size = 200
method = "aec"  # "vae"
model_dims = 32
version = "v2p9"  # "v2v2" "v2p9"

dm.proc_type = "skl"
dm.modal.images_glob = f"ang*rfl/ang*_rfl_{version}/ang*_corr_{version}_img"
TileManager.block_size = block_size
TileManager.block_index = [0, 5]
AvirisDataManager.version = version
dm.modal.valid_aviris_bands = [[5, 193], [214, 283], [319, 10000]]
dm.modal.model_dims = model_dims
dm.modal.reduce_method = method
dm.modal.reduce_nepoch = 3
dm.modal.reduce_focus_nepoch = 10
dm.modal.reduce_niter = 1
dm.modal.reduce_focus_ratio = 10.0
dm.modal.reduce_dropout = 0.0
dm.modal.reduce_learning_rate = 1e-4
dm.modal.refresh_model = False
dm.modal.modelkey = f"b{block_size}.{method}"

dm.loadCurrentProject()
classes = [('Class-1', "cyan"),
           ('Class-2', "green"),
           ('Class-3', "magenta"),
           ('Class-4', "blue")]
lm().setLabels(classes)

block0: Block = tm().getBlock()
pdata0: xa.DataArray = block0.getPointData()[0]
print( pdata0.mean() )
model_data = dm.getModelData( )
print( model_data.shape )

dm.modal.set_current_image(image_index)
block1 = tm().getBlock()
pdata1: xa.DataArray = block1.getPointData()[0]
print( pdata1.mean() )
model_data = dm.getModelData( )
print( model_data.shape )
