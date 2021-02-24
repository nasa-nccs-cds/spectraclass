import math, random, numpy as np
import matplotlib.pyplot as plt
from spectraclass.test.texture.util import *
from spectraclass.features.texture.manager import TextureManager, texm
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.gui.spatial.application import Spectraclass
from spectraclass.data.spatial.tile.tile import Block
import xarray as xa

t0 = time.time()
dm: DataManager = DataManager.initialize("demo1",'keelin')
app = Spectraclass.instance()
project_data: xa.Dataset = dm.loadCurrentProject("main")
print(f"Loaded project data: vars = {project_data.variables.keys()}")
block: Block = tm().getBlock()
nF = block.data.shape[0]
input_image: np.ndarray = block.data.data
print(f" Loaded block {tm().block_index} in time {time.time() - t0} sec, shape = {input_image.shape} ")

gabor_features: np.ndarray = texm().gabor_features( input_image )

fig, axs = plt.subplots( 2, nF )
for iF in range(nF):
    plot2( axs, 0, iF, input_image[iF], f"Band-{iF}" )
    plot2( axs, 1, iF, gabor_features[iF], f"GF-{iF}")
plt.show()