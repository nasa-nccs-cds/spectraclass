nfeatures = 64
import tensorflow as tf
import numpy as np
import xarray as xa
from spectraclass.model.labels import lm
from spectraclass.learn.models.spatial import SpatialModelWrapper
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.data.spatial.tile.tile import Block, Tile

dm: DataManager = DataManager.initialize( "img_mgr", 'aviris' )
location = "desktop"
if location == "adapt":
    dm.modal.cache_dir = "/adapt/nobackup/projects/ilab/cache"
    dm.modal.data_dir = "/css/above/daac.ornl.gov/daacdata/above/ABoVE_Airborne_AVIRIS_NG/data/"
elif location == "desktop":
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
else: raise Exception( f"Unknown location: {location}")

block_size = 150
method = "aec" # "vae"
model_dims = 32
version = "v2v2"
month = "201908"

dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris/adapt"
dm.modal.ext =  "_img"
dm.use_model_data = True
dm.proc_type = "skl"
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
            ('Class-4', "blue")]

lm().setLabels( classes )

input_shape = SpatialModelWrapper.get_input_shape()
nclasses = lm().nLabels
ks =  3

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=input_shape))
model.add(tf.keras.layers.Conv2D(nfeatures, (ks, ks), activation='relu', padding="same"))
model.add(tf.keras.layers.Reshape(SpatialModelWrapper.flatten(input_shape, nfeatures)))
model.add(tf.keras.layers.Dense(nfeatures, activation='relu'))
model.add(tf.keras.layers.Dense(nclasses, activation='softmax'))

block: Block = tm().getBlock()
bdata: xa.DataArray = block.data.transpose('y','x','band').fillna(0.0)
input_batch: np.ndarray = bdata.expand_dims('batch',0).values

print( input_batch.shape )
