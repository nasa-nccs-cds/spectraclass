import random, numpy as np
import matplotlib.pyplot as plt
from pywt import dwt2
from spectraclass.test.texture.util import *
from skimage.filters.rank import entropy
from skimage.morphology import disk
from spectraclass.test.texture.util import load_test_data

fig, axs = plt.subplots( 2, 3 )

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iLayer = 0
wavelet_type = 'db1'

texband: np.ndarray = load_test_data( dataset_type, dsid, data_type, iLayer ).data
cA, (cH, cV, cD) = dwt2( texband, wavelet_type )
E = (cH ** 2 + cV ** 2 + cD ** 2)

plot( axs, 0, texband, f"CHR Band {iLayer}" )
plot( axs, 1, cA, f"cA" )
plot( axs, 2, cH, f"cH" )
plot( axs, 3, cV, f"cV" )
plot( axs, 4, cD, f"cD" )
plot( axs, 5, E,  f"E" )

plt.show()