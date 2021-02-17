import random, numpy as np
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
import matplotlib.pyplot as plt
import scipy.stats as stats
from skimage.feature import texture as sktex
import time, xarray as xa
from skimage.morphology import disk, ball
from skimage.filters.rank import windowed_histogram

t0 = time.time()
dm: DataManager = DataManager.initialize("demo1",'keelin')
project_data: xa.Dataset = dm.loadCurrentProject("main")
block = tm().getBlock()
reduced: np.ndarray = project_data.reduction.data.transpose().reshape( project_data.model.size, *block.shape )
t1 = time.time()
print( f"Loaded data in {t1-t0} sec")

R = 3
P = 8 * R
nb = 1
b0 = 6
method = "uniform"  # "var" "ror" "uniform"
dset = "raw"
fig = plt.figure()
hist_neighborhood = 5

for ib in range(nb):
    texband: np.ndarray = block.data.data[ib+b0] if dset == "raw" else reduced[ib+b0]
    lbp: np.ndarray = sktex.local_binary_pattern( texband, P, R, method )
    local_hist: np.ndarray = windowed_histogram(  lbp, disk( hist_neighborhood )  )

    a0=fig.add_subplot(nb,2,ib*2+1)
    imgplot0 = plt.imshow( texband )
    imgplot0.set_cmap('nipy_spectral')

    a1=fig.add_subplot(nb,2,ib*2+2)
    imgplot1 = plt.imshow(lbp)
    imgplot1.set_cmap('nipy_spectral')

plt.show()