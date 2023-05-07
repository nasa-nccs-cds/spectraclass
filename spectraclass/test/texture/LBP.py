import os, time, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import xarray as xa
from spectraclass.features.texture.lbp import LBP
from spectraclass.test.texture.util import load_test_data, plot

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iBand = 0
nFeatures = 1
hist_radius = 3
R = [ 1.0, 2.0, 3.0 ]

texband: xa.DataArray = load_test_data( dataset_type, dsid, data_type, iBand )

lbp = LBP( nfeat=nFeatures, R = R, hist_radius=hist_radius )
features: List[ np.ndarray ] = lbp.compute_band_features( texband.data )

fig, axs = plt.subplots( 1, nFeatures+1 )
plot( axs, 0, texband.data, f"image {data_type}-{iBand}" )
for iF in range( nFeatures ): plot( axs, iF+1, features[iF], f"LBP-F{iF}" )

plt.show()
