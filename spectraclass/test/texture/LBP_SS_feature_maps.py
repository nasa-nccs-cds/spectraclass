import os, time, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import scipy.stats as stats
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.axes import Axes, BarContainer
from sklearn.metrics.pairwise import chi2_kernel
import xarray as xa
from skimage.feature import texture as sktex
from skimage.morphology import disk, ball
from skimage.filters.rank import windowed_histogram
from spectraclass.test.texture.util import load_test_data

fig, axs = plt.subplots( 2, 4 )
def plot( iP: int, data: np.ndarray, title: str ) -> AxesImage:
    ax = axs[iP//4,iP%4]
    ax.set_yticks([]); ax.set_xticks([])
    ax.title.set_text( title )
    imgplot: AxesImage = ax.imshow(data)
    imgplot.set_cmap('nipy_spectral')
    return imgplot

dataset_type = "chr"
dsid = "ks"
data_type = "pca"
iBand = 0
texband: xa.DataArray = load_test_data( dataset_type, dsid, data_type, iBand )

R = 1.5
P = round( 8 * R )
method = "uniform"  # "var" "ror" "uniform"
hist_disk_radius = 2

lbp: np.ndarray = sktex.local_binary_pattern( texband.data, P, R, method ).astype(np.int)
print( f" calculated LBP, shape = {lbp.shape}, range = [ {lbp.min()}, {lbp.max()} ] ")

hist_array: np.ndarray = windowed_histogram( lbp, disk( hist_disk_radius )  )
[ys,xs,nbin] = hist_array.shape
print( f" calculated hist_array, shape = {hist_array.shape}")

t0 = time.time()
K: np.ndarray = chi2_kernel( hist_array.reshape([ys*xs,nbin]), gamma=.5 )
dt = time.time() - t0
print( f"Computed Kernel in {dt} sec ({dt/60} min), K shape = {K.shape}, K range = [ {K.min()}, {K.max()} ]")

transformer = KernelPCA(n_components=8, kernel='precomputed')
reduced = transformer.fit_transform(K)

plot( 0, texband.data, f"{data_type}-{iBand}" )
plot( 1, lbp, f"LBP: P={P}, R={R}" )
plt.show()
