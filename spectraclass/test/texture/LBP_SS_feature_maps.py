import os, time, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import matplotlib.pyplot as plt
import scipy.stats as stats
from pynndescent import NNDescent
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.axes import Axes, BarContainer
import xarray as xa
from skimage.feature import texture as sktex
from skimage.morphology import disk, ball
from skimage.filters.rank import windowed_histogram
from spectraclass.test.texture.util import load_test_data

def getProbabilityGraph( data: np.ndarray, nneighbors: int ) -> NNDescent:    # data: array, shape = (n_samples, n_features)
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, 2 * int(round(np.log2(data.shape[0]))))
    kwargs = dict(n_trees=n_trees, n_iters=n_iters, n_neighbors=nneighbors, max_candidates=60, verbose=True, metric="wasserstein")
    return  NNDescent(data, **kwargs)

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
n_graph_neighbors = 5

t0 = time.time()
lbp: np.ndarray = sktex.local_binary_pattern( texband.data, P, R, method ).astype(np.int)
print( f" calculated LBP in time {time.time()-t0} sec, shape = {lbp.shape}, range = [ {lbp.min()}, {lbp.max()} ] ")

t1 = time.time()
hist_array: np.ndarray = windowed_histogram( lbp, disk( hist_disk_radius )  )
[ys,xs,nbin] = hist_array.shape
print( f" calculated hist_array in time {time.time()-t1} sec, shape = {hist_array.shape}")

t2 = time.time()
graph = getProbabilityGraph( hist_array.reshape( [ys*xs, nbin] ), n_graph_neighbors )
print( f" calculated ProbabilityGraph in time {time.time()-t2} sec")

plot( 0, texband.data, f"{data_type}-{iBand}" )
plot( 1, lbp, f"LBP: P={P}, R={R}" )
plt.show()
