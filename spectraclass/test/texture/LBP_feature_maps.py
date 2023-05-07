import os, time, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import scipy.stats as stats
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

lbp: np.ndarray = sktex.local_binary_pattern( texband.data, P, R, method ).astype(np.int32)
print( f" calculated LBP, shape = {lbp.shape}, range = [ {lbp.min()}, {lbp.max()} ] ")

hist_array: np.ndarray = windowed_histogram( lbp, disk( hist_disk_radius )  )
[ys,xs] = hist_array.shape[:2]
print( f" calculated hist_array, shape = {hist_array.shape}")

plot( 0, texband.data, f"{data_type}-{iBand}" )
plot( 1, lbp, f"LBP: P={P}, R={R}" )

def get_stats( local_hist: np.ndarray ) -> np.ndarray:
    dpd: stats.rv_discrete =  stats.rv_discrete( values=( np.arange(local_hist.size), local_hist ) )
    moments: List[float] = [ xm.tolist() for xm in dpd.stats( moments='mvsk' ) ]
    moments.append( dpd.entropy().tolist() )
    return np.array( moments )

t0 = time.time()
hdata: np.ndarray = hist_array.reshape( [ xs * ys, hist_array.shape[2] ] )
hdata_stats: np.ndarray = np.apply_along_axis( get_stats, 1, hdata )
nf = hdata_stats.shape[1]
hdata_stats = hdata_stats.reshape( [ ys, xs, nf ] )
dt = time.time() - t0

print( f"Computed tex-stats in {dt} sec ({dt/60} min), stats shape = {hdata_stats.shape}, hdata shape = {hdata.shape}")
for iT in range( hdata_stats.shape[2] ):
    plot( iT+2, hdata_stats[:,:,iT], f"TF-{iT}" )

plt.show()
