import math, random, numpy as np
import matplotlib.pyplot as plt
from spectraclass.test.texture.util import *
from spectraclass.test.texture.util import load_test_data
from spectraclass.features.texture.gabor import Gabor
sr2 = math.sqrt( 2.0 )
nFeatures = 1
nBands = 7

gabor = Gabor( nfeat=nFeatures )

fig, axs = plt.subplots( nFeatures+1, nBands )
for iB in range(nBands):
    image: np.ndarray = load_test_data( "chr", "ks", "raw", iB ).data
    print( f"\nLoaded raw image, band = {iB}, shape = {image.shape}, range = {[ image.min(), image.max() ]}")

    t0 = time.time()
    features = gabor.compute_band_features(image)
    print(f"Computed {nFeatures} Gabor features in time = {time.time()-t0} sec")

    plot2( axs, 0, iB, image, f"Band-{iB}" )
    for iF, feature in enumerate(features):
        plot2( axs, iF+1, iB, feature, f"GF-{iF}" )

plt.show()