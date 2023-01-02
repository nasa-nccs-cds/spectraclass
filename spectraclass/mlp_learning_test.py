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
from spectraclass.learn.mlp import MLP

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
dm.use_model_data = False
dm.proc_type = "skl"

tm.block_size = block_size
tm.block_index = [0,0]
dm.modal.version = version
dm.modal.refresh_model = False
dm.modal.modelkey = f"b{block_size}.aec"

dm.loadCurrentProject()
ClassificationManager.nepochs = 50
ClassificationManager.mid = "mlp"
nclasses = lm().nLabels

model = MLP( 'mlp', layers = [32, 16] )
cm().addNetwork(model)

for ic in [1,2,3,4]:
    lm().addMarker( Marker( "test", [100*ic], ic ) )

cm().learn_classification()
cm().apply_classification()

