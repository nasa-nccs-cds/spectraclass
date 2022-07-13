from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.data.spatial.tile.tile import Block, Tile
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.learn.manager import ClassificationManager, cm
from spectraclass.learn.cnn import CNN
from spectraclass.model.labels import lm
import numpy as np
from spectraclass.learn.base import KerasModelWrapper, LearningModel
from spectraclass.learn.base import LearningModel
import os, xarray as xa
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

labels_file = "/Volumes/Shared/Cache/spectraclass/aviris/img_mgr/labels/20170720t004130_b-128-1-4-raw-labels.nc"
dm.modal.ext = "_img"
dm.proc_type = "cpu"
TileManager.block_size = 128
TileManager.block_index = [1,4]
image_index = 0
AvirisDataManager.valid_aviris_bands = [[5, 193], [214, 283], [319, 10000]]
AvirisDataManager.reduce_method = None
nfeatures = 32

classes = [('Class-1', "cyan"),
           ('Class-2', "green"),
           ('Class-3', "magenta"),
           ('Class-4', "blue")]

dm.loadCurrentProject()
lm().setLabels(classes)

block: Block = tm().getBlock()
bdata: xa.DataArray = block.data.transpose('y','x','band').fillna(0.0)

labels_dset: xa.Dataset = xa.open_dataset( labels_file )
labels_name = f"labels-{image_index}-{TileManager.block_index[0]}-{TileManager.block_index[1]}"
labels_array: xa.DataArray = labels_dset[labels_name]  #

print( f"Build CNN, input shape = {bdata.shape}" )
cnn =  CNN.build( bdata.shape, nfeatures, lm().nLabels )
model = KerasModelWrapper("cnn",cnn)
input_batch: np.ndarray = bdata.expand_dims('batch',0).values
result = model.apply( input_batch )
y = np.expand_dims( LearningModel.index_to_one_hot( labels_array.values.flatten() ), 0 )
model.fit( input_batch, y )

# print(result.shape)