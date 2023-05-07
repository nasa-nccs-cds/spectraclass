import os, time, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import scipy.stats as stats
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.axes import Axes, BarContainer
import xarray as xa
from skimage.feature import texture as sktex
from skimage.morphology import disk, ball
from skimage.filters.rank import windowed_histogram
from spectraclass.test.texture.util import *

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iBand = 0
texband: xa.DataArray = load_test_data( dataset_type, dsid, data_type, iBand )

R = 1.0
P = round( 8 * R )
method = "uniform"  # "var" "ror" "uniform"
hist_disk_radius = 3
n_graph_neighbors = 5
n_features = 5
plot_start_band = 0
n_reduce_epochs = 70
rep_tst_bnd = 0
loss = "mean_squared_error"
fig, axs = plt.subplots( 2, n_features )

t0 = time.time()
lbp: np.ndarray = sktex.local_binary_pattern( texband.data, P, R, method ).astype(np.int32)
print( f" calculated LBP in time {time.time()-t0} sec, shape = {lbp.shape}, range = [ {lbp.min()}, {lbp.max()} ] ")

t1 = time.time()
hist_array: np.ndarray = windowed_histogram( lbp, disk( hist_disk_radius )  )
[ys,xs,nbin] = hist_array.shape
sample_probabilties: np.ndarray = hist_array.reshape( [ys*xs, nbin] )
print( f" calculated hist_array in time {time.time()-t1} sec, shape = {hist_array.shape}, norm[:10] = {sample_probabilties.sum(axis=1)[:10]}")

t1 = time.time()
( tex, rep ) = autoencoder_reduction( sample_probabilties, n_features, n_reduce_epochs, loss=loss  )
texture_features = tex.reshape( [ys,xs,n_features] )
reproduction = rep.reshape( [ys,xs,nbin] )
print( f" calculated autoencoder reduction in time {time.time()-t1} sec, shape = {texture_features.shape}")

for iB in range( 0, n_features ):
    band = iB + plot_start_band
    plot( axs, iB,               hist_array[:,:,band],  f"LBP-Hist[{band}]" )
    plot( axs, iB+n_features,  reproduction[:,:,band],  f"Reproduction[{band}]" )
plt.show()