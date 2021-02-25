import os, time, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import matplotlib.pyplot as plt
import xarray as xa
from spectraclass.features.texture.lbp import LBP
from spectraclass.test.texture.util import load_test_data, plot

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iBand = 0
nFeatures = 4
R = 1.5

texband: xa.DataArray = load_test_data( dataset_type, dsid, data_type, iBand )

lbp = LBP( nfeat=nFeatures, R = R )
features: List[ np.ndarray ] = lbp.compute_band_features( texband.data )

fig, axs = plt.subplots( 2, 3 )
plot( axs, 0, texband.data, f"image {data_type}-{iBand}" )
for iF in range( nFeatures ): plot( axs, iF+1, features[iF], f"LBP-F{iF}" )

plt.show()
