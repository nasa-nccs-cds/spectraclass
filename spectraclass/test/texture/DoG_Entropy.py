import random, numpy as np
from spectraclass.test.texture.util import load_test_data
import matplotlib.pyplot as plt
from skimage.filters import difference_of_gaussians
from skimage.filters.rank import entropy
from skimage.morphology import disk
from spectraclass.test.texture.util import scale
from scipy import ndimage as ndi

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iLayer = 0
texband: np.ndarray = load_test_data( dataset_type, dsid, data_type, iLayer ).data

nb = 4
mode = "nearest"  # ‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’
dset = ( "raw", 3 )
sigma0: float = 1.0
sigma1: float = 3.0
edisk_size: int = 4
ni = 3

fig = plt.figure()
for ib in range(nb):

    sigma1 = sigma0 + ( .5 *  (ib+1) )
    input: np.ndarray =  difference_of_gaussians( texband, sigma0, sigma1, mode=mode )
    scaled_input = scale(input)
    print( [ scaled_input.max(), scaled_input.min() ] )
    entropy_img: np.ndarray = entropy( scaled_input, disk( edisk_size ) )
    plots =  [ texband, scaled_input, entropy_img ]
    titles = [ f"{dset[0]}-data[{dset[1]}]", f"DoG: S0={sigma0:.2f}, S1={sigma1:.2f}", f"Entropy: D={edisk_size}" ]

    for ii in range(ni):
        ax = fig.add_subplot( nb, ni, ib*ni + ii+1 )
        ax.set_yticks([]); ax.set_xticks([])
        ax.title.set_text( titles[ii] )
        imgplot0 = plt.imshow( plots[ii] )
        imgplot0.set_cmap('nipy_spectral')

plt.tight_layout( pad=0.01, h_pad=0.01, w_pad=0.01 )
plt.show()