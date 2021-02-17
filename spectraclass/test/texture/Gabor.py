import random, numpy as np
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy
from skimage.morphology import disk
import time, xarray as xa
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
method = "uniform"  # "var" "ror" "uniform"
dset = ( "raw", 3 )
sigma: float = 3
frequency: float = 0.25 # (0.05, 0.25)
theta: float = np.pi / 4.0
edisk_size: int = 5
fig = plt.figure()

for ib in range(nb):

    sigma = 1.5 + (.5 *  ib)
    texband: np.ndarray = block.data.data[dset[1]] if dset[0] == "raw" else reduced[dset[1]]
    kernel: np.ndarray = np.real( gabor_kernel( frequency, theta=theta, sigma_x=sigma, sigma_y=sigma ) )
    filtered: np.ndarray =  np.sqrt( ndi.convolve(texband, np.real(kernel), mode='reflect')**2 + ndi.convolve(texband, np.imag(kernel), mode='reflect')**2, dtype=np.float )
    entropy_img: np.ndarray = entropy( filtered/filtered.max(), disk( edisk_size ) )
    plots = ( texband, filtered, entropy_img )
    titles = ( f"{dset[0]}-data[{dset[1]}]", f"Gabor: F={frequency:.2f}, T={theta:.2f}, S={sigma:.2f}", f"Entropy: D={edisk_size}")

    for ii in range(ni):
        ax = fig.add_subplot( nb, ni, ib*ni + ii+1 )
        ax.set_yticks([]); ax.set_xticks([])
        ax.title.set_text( titles[ii] )
        imgplot0 = plt.imshow( plots[ii] )
        imgplot0.set_cmap('nipy_spectral')

plt.tight_layout( pad=0.01, h_pad=0.01, w_pad=0.01 )
plt.show()