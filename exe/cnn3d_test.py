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

def count_nan( array: np.ndarray ) -> int:
    return np.count_nonzero( np.isnan(array) )

def osize( isize: int, ks: int, s: int ) -> int:
    return ((isize-ks)//s) + 1

dm: DataManager = DataManager.initialize( "img_mgr", 'aviris' )
location = "laptop"
version = "v2p9"  # "v2v2" "v2p9"
month = "201707" # "201707" "201908"

if location == "adapt":
    dm.modal.cache_dir = "/adapt/nobackup/projects/ilab/cache"
    dm.modal.data_dir = "/css/above/daac.ornl.gov/daacdata/above/ABoVE_Airborne_AVIRIS_NG/data/"
    dm.modal.images_glob = f"ang{month}*rfl/ang*_rfl_{version}/ang*_corr_{version}_img"
elif location == "desktop":
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
    dm.modal.images_glob = f"ang{month}*rfl/ang*_rfl_{version}/ang*_corr_{version}_img"
elif location == "laptop":
    dm.modal.cache_dir = "/Volumes/archive/Cache"
    dm.modal.data_dir = "/Volumes/Shared/Data/Aviris/IndianPines/aviris_hyperspectral_data"
    dm.modal.images_glob = "*_AVIRIS_IndianPine_Site3.tif"
else: raise Exception( f"Unknown location: {location}")

block_size = 200
method = "aec" # "vae"
model_dims = 32
dm.use_model_data = True
dm.proc_type = "skl"

TileManager.block_size = block_size
TileManager.block_index = [0,5]
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
ks =  5
nfeatures = 8
strides = 3
device = 'cpu'
CNN1 = tf.keras.layers.Conv3D( filters=nfeatures, kernel_size=ks, activation='relu', padding="valid", strides=strides )

block: Block = tm().getBlock()
spatial_data: xa.DataArray = block.getSpectralData( raster=True )

with tf.device(f'/{device}:0'):
    print(f"training_set shape = {spatial_data.shape}")
    nb = spatial_data.shape[1]
    input: tf.Tensor = tf.convert_to_tensor( spatial_data.values )
    result1: tf.Tensor = CNN1( input )
    nb = osize(nb,ks,strides)
    print(f"result1 shape = {result1.shape}, nb = {nb}")
