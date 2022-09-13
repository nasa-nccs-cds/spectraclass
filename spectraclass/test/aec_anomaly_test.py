import pickle, random, time, numpy as np
from spectraclass.data.base import DataManager
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.reduction.embedding import rm
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Optional, Dict, Tuple
import os, xarray as xa
rng = np.random.default_rng()

def pnorm( data: xa.DataArray, dim: int = 1 ) -> xa.DataArray:
    dave, dmag = np.nanmean(data.values, keepdims=True, axis=dim), np.nanstd(data.values, keepdims=True, axis=dim)
    normed_data = (data.values - dave) / dmag
    return data.copy(data=normed_data)

dm: DataManager = DataManager.initialize( 'img_mgr', "aviris" )
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"

block_size = 150
method = "aec"  # "vae"
model_dims = 32
nepochs = 20
use_saved_weights = True
nbins = 64

dm.modal.ext = "_img"
dm.use_model_data = True
dm.proc_type = "skl"
TileManager.block_size = block_size
TileManager.block_index = [0, 7]
AvirisDataManager.valid_aviris_bands = [[5, 193], [214, 283], [319, 10000]]
AvirisDataManager.model_dims = model_dims
AvirisDataManager.reduce_method = method
AvirisDataManager.reduce_nepoch = nepochs
AvirisDataManager.reduce_niter = 1
AvirisDataManager.refresh_model = not use_saved_weights
AvirisDataManager.modelkey = f"b{block_size}.{method}"

block: Block = tm().getBlock()
point_data, grid = block.getPointData()
norm_data: xa.DataArray = pnorm( point_data )

encoder_result, reproduced_data = dm.modal.autoencoder_reduction( norm_data )

anomaly: np.ndarray = np.abs( norm_data.values - reproduced_data ).sum( axis=-1, keepdims=False )
dmask: np.ndarray = (anomaly > 0.0)
hist, edges = np.histogram( anomaly[dmask], nbins )
print( f" ----> ANOMALY: shape = {anomaly.shape}, range = [{anomaly.min(where=dmask ,initial=np.inf)},{anomaly.max()}], edges shape = {edges.shape} ")

x: np.ndarray = np.arange( nbins )
counts: np.ndarray = np.cumsum(hist)
ti: int = np.abs(counts-counts[-1]*0.75).argmin()
t: float = edges[ti+1]
amask: np.ndarray = (anomaly > t)
nanom: int = np.count_nonzero(amask)
print( f" ti = {ti}, counts[t] = {counts[ti]}, f = {counts[ti]/counts[-1]}, t={t}")
print( f" amask: shape = {amask.shape}, nanom = {nanom}, f = {nanom/amask.size}")

anom_data = norm_data[ amask ]
std_data = norm_data[ ~amask ]
std_data_sample = rng.choice( std_data, anom_data.shape[0], replace=False, axis=0, shuffle=False )
new_data = np.concatenate((anom_data, std_data_sample), axis=0)
print( f" anom_data: shape = {anom_data.shape}, std_data: shape = {std_data.shape}, std_data_sample: shape = {std_data_sample.shape}, new_data: shape = {new_data.shape}")

fig, ax = plt.subplots( 1, 1 )
plt.plot( x, hist )
plt.show()