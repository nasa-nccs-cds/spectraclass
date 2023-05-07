import math, random, numpy as np
from spectraclass.test.texture.util import *
from spectraclass.test.texture.util import load_test_data
from spectraclass.features.texture.lbp import LBP

nFeatures = 1
hist_radius = 3
reduce_method = "pca"
R = [ 1.0, 2.0, 3.0 ]
nBands = 7

lbp = LBP( nfeat=nFeatures, R = R, hist_radius=hist_radius, reduce_method=reduce_method )

fig, axs = plt.subplots( nFeatures+1, nBands )
for iB in range(nBands):
    image: np.ndarray = load_test_data( "chr", "ks", "raw", iB ).data
    print( f"\nLoaded raw image, band = {iB}, shape = {image.shape}, range = {[ image.min(), image.max() ]}")

    t0 = time.time()
    features = lbp.compute_band_features(image)
    print(f"Computed {nFeatures} Gabor features in time = {time.time()-t0} sec")

    plot2( axs, 0, iB, image, f"Band-{iB}" )
    for iF, feature in enumerate(features):
        plot2( axs, iF+1, iB, feature, f"GF-{iF}" )

plt.show()