import os, time, random, numpy as np
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
from spectraclass.test.texture.util import *

fig, axs = plt.subplots( 2, 4 )

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iBand = 0
texband: xa.DataArray = load_test_data( dataset_type, dsid, data_type, iBand )

R = 1.0
P = 4
method = "uniform"  # "var" "ror" "uniform"
hist_disk_radius = 5
n_features = 3
n_reduce_epochs = 100
rep_tst_bnd = 0

t0 = time.time()
lbp: np.ndarray = sktex.local_binary_pattern( texband.data, P, R, method ).astype(np.int)
print( f" calculated LBP in time {time.time()-t0} sec, shape = {lbp.shape}, range = [ {lbp.min()}, {lbp.max()} ] ")

t1 = time.time()
hist_array: np.ndarray = windowed_histogram( lbp, disk( hist_disk_radius )  )
[ys,xs,nbin] = hist_array.shape
sample_probabilties: np.ndarray = hist_array.reshape( [ys*xs, nbin] )
print( f" calculated hist_array in time {time.time()-t1} sec, shape = {hist_array.shape}, norm[:10] = {sample_probabilties.sum(axis=1)[:10]}")

t1 = time.time()
( tex, rep ) = autoencoder_reduction( sample_probabilties, n_features, n_reduce_epochs  )
texture_features = tex.reshape( [ys,xs,n_features] )
reproduction = rep.reshape( [ys,xs,nbin] )
print( f" calculated autoencoder reduction in time {time.time()-t1} sec, shape = {texture_features.shape}")

plot( axs, 0, texband.data, f"{data_type}-{iBand}" )
plot( axs, 1, hist_array[:,:,rep_tst_bnd], f"LBP Hist: P={P}, R={R}, r={hist_disk_radius}" )
plot( axs, 2, reproduction[:,:,rep_tst_bnd], f"Reproduction using {n_features}-D coding" )

for iF in range( n_features ):
    texture_image = texture_features[:,:,iF]
    plot( axs, iF+3, texture_image, f"TF={iF}" )
plt.show()





# t2 = time.time()
# n_graph_neighbors = 5
# nngraph: NNDescent = getProbabilityGraph( sample_probabilties, n_graph_neighbors )
# (indices, distances) = nngraph.neighbor_graph
# print( f" calculated ProbabilityGraph in time {time.time()-t2} sec, shapes = {indices.shape} {distances.shape}")
#
# t3 = time.time()
# mapper = UMAP.instance( metric="hellinger", n_components=3, n_neighbors=n_graph_neighbors, init="random" )
# embedding = mapper.embed( sample_probabilties, nngraph=nngraph )
# print( f" calculated UMAP embedding in time {time.time()-t3} sec, shape = {embedding.shape}")
