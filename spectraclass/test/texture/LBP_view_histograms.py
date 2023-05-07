import os, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import scipy.stats as stats
import xarray as xa
from skimage.feature import texture as sktex
from skimage.morphology import disk, ball
from skimage.filters.rank import windowed_histogram
from spectraclass.test.texture.util import load_test_data

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iBand = 0
texband: xa.DataArray = load_test_data( dataset_type, dsid, data_type, iBand )

R = 1.0
P = round( 8 * R )
method = "uniform"  # "var" "ror" "uniform"
hist_disk_radius = 3.0

fig: Figure = plt.figure()
grid = plt.GridSpec(3, 3, wspace=0.1, hspace=0.1)

lbp: np.ndarray = sktex.local_binary_pattern( texband.data, P, R, method ).astype(np.int32)
hist_array: np.ndarray = windowed_histogram( lbp, disk( hist_disk_radius )  )
sample_hist = hist_array[ 0, 0 ]

ax0: Axes = plt.subplot( grid[ 0:2, 0:2] )
ax0.set_yticks([]); ax0.set_xticks([])
ax0.title.set_text(f"{data_type}-{iBand}")

ax1: Axes = plt.subplot( grid[ 0:2, 2:4 ] )
ax1.set_yticks([]); ax1.set_xticks([])
ax1.title.set_text(f"LBP: P={P}, R={R}")

axH: Axes = plt.subplot( grid[2,:] )
axH.set_ylim( 0.0, 1.0 ); axH.set_xlim( auto=True )

imgplot0: AxesImage = ax0.imshow( texband.data )
imgplot0.set_cmap('nipy_spectral')

imgplot1: AxesImage = ax1.imshow( lbp )
imgplot1.set_cmap('nipy_spectral')

imgplotH: BarContainer = axH.bar( range(sample_hist.size), sample_hist, align="edge" )

def onMouseClick( event ):
    x, y = int(event.xdata), int(event.ydata)
    local_hist = hist_array[ y, x ]
#    cpd: stats.rv_continuous = stats.rv_histogram( ( local_hist, range( local_hist.size+1 ) ) )
    dpd: stats.rv_discrete =  stats.rv_discrete( values=( np.arange(local_hist.size), local_hist ) )
    cmoments  = [ xm.tolist() for xm in dpd.stats( moments='mvsk' ) ]
    print(f"Plotting Histogram at (x,y) = {[x, y]}, shape = {local_hist.shape}, entropy = {dpd.entropy()}, stats = {cmoments}")
    for rect,h in zip(imgplotH,local_hist): rect.set_height(h)

    fig.canvas.draw()

imgplot1.figure.canvas.mpl_connect( 'button_press_event', onMouseClick )
plt.show()
