import random, numpy as np
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
import matplotlib.pyplot as plt
from skimage.filters import difference_of_gaussians
from skimage.filters.rank import entropy
from skimage.morphology import disk
import time, xarray as xa
from spectraclass.test.texture.util import scale
from scipy import ndimage as ndi

t0 = time.time()
dm: DataManager = DataManager.initialize("demo1",'keelin')
project_data: xa.Dataset = dm.loadCurrentProject("main")
block = tm().getBlock()
reduced: np.ndarray = project_data.reduction.data.transpose().reshape( project_data.model.size, *block.shape )
t1 = time.time()
print( f"Loaded data in {t1-t0} sec")

nb = 4
ni = 3
mode = "nearest"  # ‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’
dset = ( "raw", 3 )
sigma0: float = 1.0
sigma1: float = 3.0
edisk_size: int = 5
fig = plt.figure()

for ib in range(nb):

    sigma1 = sigma0 + ( .5 *  (ib+1) )
    texband: np.ndarray = block.data.data[dset[1]] if dset[0] == "raw" else reduced[dset[1]]
    filtered: np.ndarray =  difference_of_gaussians( texband, sigma0, sigma1, mode=mode )
    nfilt = scale(filtered)
    print( [ nfilt.max(), nfilt.min() ] )
    entropy_img: np.ndarray = entropy( nfilt, disk( edisk_size ) )
    plots = ( texband, filtered, entropy_img )
    titles = ( f"{dset[0]}-data[{dset[1]}]", f"DoG: S0={sigma0:.2f}, S1={sigma1:.2f}", f"Entropy: D={edisk_size}")

    for ii in range(ni):
        ax = fig.add_subplot( nb, ni, ib*ni + ii+1 )
        ax.set_yticks([]); ax.set_xticks([])
        ax.title.set_text( titles[ii] )
        imgplot0 = plt.imshow( plots[ii] )
        imgplot0.set_cmap('nipy_spectral')

plt.tight_layout( pad=0.01, h_pad=0.01, w_pad=0.01 )
plt.show()