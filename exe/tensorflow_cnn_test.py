nfeatures = 64
import tensorflow as tf
import numpy as np
import xarray as xa
from spectraclass.learn.base import LearningModel
from spectraclass.learn.models.spatial import SpatialModelWrapper
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.data.spatial.tile.tile import Block, Tile
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.model.labels import LabelsManager, Action, lm

def get_training_set( nclasses ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block: Block = tm().getBlock()
    base_data: xa.DataArray = block.getModelData(True)
    tdims = [base_data.dims[1], base_data.dims[2], base_data.dims[0]]
    training_data: np.ndarray = base_data.transpose(*tdims).fillna(0.0).expand_dims('batch', 0).values
    grid_size = training_data.shape[1]*training_data.shape[2]
    labels: np.ndarray = np.random.randint( 0, nclasses, [grid_size] )
    training_labels = np.expand_dims(LearningModel.index_to_one_hot(labels), 0)
    label_mask = np.expand_dims( (labels > 0), 0 )
    sample_weights: np.ndarray = np.where(label_mask, 1.0, 0.0)
    test_mask = np.full(label_mask.shape, False)
    return (training_data, training_labels, sample_weights, test_mask )

dm: DataManager = DataManager.initialize( "img_mgr", 'aviris' )
location = "explore"
if location == "explore":
    dm.modal.cache_dir = "/explore/nobackup/projects/ilab/cache"
    dm.modal.data_dir = "/css/above/daac.ornl.gov/daacdata/above/ABoVE_Airborne_AVIRIS_NG/data/"
elif location == "desktop":
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris/adapt"
else: raise Exception( f"Unknown location: {location}")

block_size = 150
method = "aec" # "vae"
model_dims = 32
version = "v2v2"
month = "201908"

dm.modal.ext =  "_img"
dm.use_model_data = True
dm.proc_type = "cpu"
dm.modal.images_glob = f"ang{month}*rfl/ang*_rfl_{version}/ang*_corr_{version}_img"
TileManager.block_size = block_size
TileManager.block_index = [1,7]
AvirisDataManager.version = version
dm.modal.valid_aviris_bands = [ [5,193], [214,283], [319,10000] ]
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
classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue") ]

lm().setLabels( classes )

input_shape = SpatialModelWrapper.get_input_shape()
nclasses = lm().nLabels
ks =  3
device = 'cpu'
nepochs = 100
opt = tf.keras.optimizers.Adam(1e-3)
loss = 'categorical_crossentropy'

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=input_shape))
model.add(tf.keras.layers.Conv2D(nfeatures, (ks, ks), activation='relu', padding="same"))
model.add(tf.keras.layers.Reshape(SpatialModelWrapper.flatten(input_shape, nfeatures)))
model.add(tf.keras.layers.Dense(nfeatures, activation='relu'))
model.add(tf.keras.layers.Dense(nclasses, activation='softmax'))
model.compile( optimizer=opt, loss=loss, metrics=['accuracy'] )

training_data, training_labels, sample_weights, test_mask = get_training_set( len(classes)+1 )
with tf.device(f'/{device}:0'):
    model.fit( training_data, training_labels, sample_weight=sample_weights, epochs=nepochs )
