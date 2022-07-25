from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
import matplotlib.pyplot as plt
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

def get_sample_weights( labels: np.ndarray, nLabels: int ) -> np.ndarray:
    sample_weights: np.ndarray = np.where( (labels == 0), 0.0, 1.0 )
    label_counts =  [ np.count_nonzero(labels==iC) for iC in range( nLabels ) ]
    label_weights = np.array( [ 1.0/lc if (lc > 0.0) else 0.0 for lc in label_counts ] )
    weights_sum = label_weights.sum()
    label_weights = { iC: label_weights[iC]/weights_sum for iC in range( nLabels )  }
    for iC in range( 1, nLabels ):
        if label_weights[iC] > 0.0:
            sample_weights[ (labels == iC) ] = label_weights[iC]
    return np.expand_dims( sample_weights, 0 )


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
nepochs = 100

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
labels_array: xa.DataArray = labels_dset[labels_name]
labels: np.ndarray = labels_array.values.flatten()

print( f"Build CNN, input shape = {bdata.shape}" )
cnn =  CNN.build( bdata.shape, nfeatures, lm().nLabels )
model = KerasModelWrapper("cnn",cnn)
input_batch: np.ndarray = bdata.expand_dims('batch',0).values
preresult: np.ndarray = model.predict( input_batch ).squeeze()

y = np.expand_dims( LearningModel.index_to_one_hot( labels ), 0 )
sample_weights: np.ndarray = get_sample_weights( labels, lm().nLabels )
model.fit( input_batch, y, sample_weight=sample_weights, nepochs=nepochs )
postresult: np.ndarray = model.predict( input_batch ).squeeze()
classresult: np.ndarray = postresult.argmax( axis=1 ).reshape(labels_array.shape)

fig, ax = plt.subplots(2,4)
ax[0,0].set_title( "Weights" )
ax[0,0].imshow( sample_weights.reshape(labels_array.shape) )
ax[0,1].set_title( "Labels" )
ax[0,1].imshow( labels_array.values )
ax[0,2].set_title( "Classification" )
ax[0,2].imshow( classresult )
ax[0,3].set_title(f"PostResult: Class 0")
ax[0,3].imshow(postresult[:, 0].reshape(labels_array.shape), cmap="jet")

for iP in range(4):
    ax[1,iP].set_title( f"PostResult: Class {iP+1}" )
    ax[1,iP].imshow( postresult[:,iP+1].reshape(labels_array.shape), cmap="jet"  )

plt.tight_layout()
plt.show()

# print(result.shape)