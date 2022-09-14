from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.application.controller import app, SpectraclassController
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.learn.manager import ClassificationManager, cm
from typing import List, Union, Tuple, Optional, Dict, Callable

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

block_size = 150
method = "aec"  # "vae"
model_dims = 32

dm.modal.ext = "_img"
dm.use_model_data = True
dm.proc_type = "skl"
TileManager.block_size = block_size
TileManager.block_index = [0, 7]
dm.modal.valid_aviris_bands = [[5, 193], [214, 283], [319, 10000]]
dm.modal.model_dims = model_dims
dm.modal.reduce_method = method
dm.modal.refresh_model = False
dm.modal.modelkey = f"b{block_size}.{method}"

dm.loadCurrentProject()
classes = [('Class-1', "cyan"),
           ('Class-2', "green"),
           ('Class-3', "magenta"),
           ('Class-4', "blue")]

lm().setLabels(classes)
dm.modal.initialize_dimension_reduction()
