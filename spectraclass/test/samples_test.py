import pickle, os
from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.tile.tile import Block
import xarray as xa, numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional, Dict, Callable

image_indices = [2,3,4]
dm: DataManager = DataManager.initialize( "demo2", 'desis' )
host = "laptop"

if (host == "laptop"):
    dm.modal.data_dir = os.path.expanduser("~/Development/Data/DESIS")
    dm.modal.cache_dir = os.path.expanduser("~/Development/Cache")
else:
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Volumes/Shared/Data/DESIS"

dm.modal.image_names = [ f"DESIS-HSI-L1C-DT0468853252_00{index}-20200628T153803-V0210" for index in image_indices ]
TileManager.block_size = 250
ModeDataManager.model_dims = 16
TileManager.block_index = [2,2]
classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

dm.loadCurrentProject()
lm().setLabels( classes )
block: Block = tm().getBlock( index=(4,4) )
(pdata, pcoords) = block.getPointData()
samples: np.ndarray = pdata.samples.values
x,y = pcoords['x'], pcoords['y']
image_data = np.full( [y.size,x.size], 0 )
cluster_gids: List[int] = pickle.load( open( "/tmp/cluster_gids.pkl", "rb" ) )
test_gid=44512

def gid2indices( gindex: int ) -> Tuple[int,int]:
    iy = gindex // x.size
    ix = gindex % x.size
    return ( iy, ix )

def gid2indices1( gindex: int ) -> Tuple[int,int]:
    iy = gindex // y.size
    ix = gindex % y.size
    return ( iy, ix )

def indices2gid( iy: int, ix: int ) -> int:
    return ix + iy * x.size

for iy in range( y.size ):
    for ix in range( x.size ):
        gid = indices2gid( iy, ix )
        image_data[iy,ix] = int( gid in samples )

for gid in cluster_gids:
    (iy,ix) = gid2indices( gid )
    image_data[iy, ix] = 2

(iy,ix) = gid2indices( test_gid )
print( f"[ys,xs] = {[y.size,x.size]}")
print( f"{test_gid} --> {ix} {iy}")
image_data[iy, ix] = 3

image = xa.DataArray( image_data, dims=['y','x'], coords = dict( x=x-x[0], y=y-y[0] ) )
image.plot.imshow( origin="upper" )
plt.show()


