import os, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.axes import Axes, BarContainer
import xarray as xa
from skimage.feature import texture as sktex
from skimage.morphology import disk, ball
from skimage.filters.rank import windowed_histogram
from spectraclass.test.texture.util import load_test_data

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iBand = 6
texband: xa.DataArray = load_test_data( dataset_type, dsid, data_type, iBand )

R = 2
P = 8 * R
method = "uniform"  # "var" "ror" "uniform"
hist_neighborhood = 5

lbp: np.ndarray = sktex.local_binary_pattern( texband.data, P, R, method ).astype(np.int)
hist_array: np.ndarray = windowed_histogram( lbp, disk( hist_neighborhood )  )
print( f" calculated hist_array, shape = {hist_array.shape}")
fig, ax = plt.subplots( 1, 5 )
ax[0].set_yticks([]); ax[0].set_xticks([])
ax[0].title.set_text(f"{data_type}-{iBand}")

ax[1].set_yticks([]); ax[1].set_xticks([])
ax[1].title.set_text(f"LBP: P={P}, R={R}")

imgplot0: AxesImage = ax[0].imshow( texband.data )
imgplot0.set_cmap('nipy_spectral')

imgplot1: AxesImage = ax[1].imshow( lbp )
imgplot1.set_cmap('nipy_spectral')

def get_stats( local_hist: np.ndarray ) -> np.ndarray:
    dpd: stats.rv_discrete =  stats.rv_discrete( values=( np.arange(local_hist.size), local_hist ) )
    return np.ndarray( [ dpd.entropy() ] + [ xm.tolist() for xm in dpd.stats( moments='mvsk' ) ] )

plt.show()
