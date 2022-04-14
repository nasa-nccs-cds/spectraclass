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

for iy in range( y.size ):
    for ix in range( x.size ):
        pid = ix + iy * x.size
        image_data[iy,ix] = int( pid in samples )

image = xa.DataArray( image_data, dims=['y','x'], coords = dict( x=x, y=y ) )
image.plot.imshow( )
plt.show()


