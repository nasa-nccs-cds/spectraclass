import tensorflow as tf
import numpy as np
import xarray as xa
from spectraclass.learn.manager import ClassificationManager, cm
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.tile import Block, Tile
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.model.labels import LabelsManager, Action, lm

def count_nan( array: np.ndarray ) -> int:
    return np.count_nonzero( np.isnan(array) )

def osize( isize: int, ks: int, s: int ) -> int:
    return ((isize-ks)//s) + 1

dm: DataManager = DataManager.initialize( "img_mgr", 'aviris' )
tm: TileManager = TileManager.instance()
classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue") ]
lm().setLabels( classes )

location = "desktop"
version = "v2p9"  # "v2v2" "v2p9"
month = "201707" # "201707" "201908"
dm.modal.valid_aviris_bands = [ [5,193], [214,283], [319,10000] ]

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
    dm.modal.valid_aviris_bands = [ [0,500], ]
else: raise Exception( f"Unknown location: {location}")

block_size = 250
dm.proc_type = "skl"

tm.block_size = block_size
tm.block_index = [0,0]
dm.modal.version = version
dm.modal.refresh_model = False
dm.modal.modelkey = f"b{block_size}.aec"

dm.loadCurrentProject()
ClassificationManager.nepochs = 5
ClassificationManager.mid = "cnn3d"
ClassificationManager.cnn_layers =[(4, 5, 3), (4, 5, 3), (3, 5, 3)]
ClassificationManager.dense_layers = [32, 16]
nclasses = lm().nLabels
use_manager = True

if use_manager:
    for ic in [1,2,3,4]:
        lm().addMarker( Marker( "test", [100*ic], ic ) )
    cm().learn_classification()
    cm().apply_classification()

else:

    ks = (5, 3, 3)
    strides = (3, 1, 1)
    device = 'cpu'
    CNN1 = tf.keras.layers.Conv3D( filters=5, kernel_size=ks, activation='relu', padding="same", strides=strides )
    CNN2 = tf.keras.layers.Conv3D( filters=4, kernel_size=ks, activation='relu', padding="same", strides=strides )
    CNN3 = tf.keras.layers.Conv3D( filters=3, kernel_size=ks, activation='relu', padding="same", strides=strides )

    block: Block = tm.getBlock()
    spatial_data: xa.DataArray = block.getSpectralData( raster=True ).expand_dims("samples",0).expand_dims("channels",4)

    with tf.device(f'/{device}:0'):
        print(f"training_set shape = {spatial_data.shape}")
        nb = spatial_data.shape[1]
        input: tf.Tensor = tf.convert_to_tensor( spatial_data.values )
        result1: tf.Tensor = CNN1( input )
        print(f"result1 shape = {result1.shape}")
        result2: tf.Tensor = CNN2( result1 )
        print(f"result2 shape = {result2.shape}")
        result3: tf.Tensor = CNN3( result2 )
        print(f"result3 shape = {result3.shape}")