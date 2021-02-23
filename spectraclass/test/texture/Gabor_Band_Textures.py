import math, random, numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.preprocessing import StandardScaler
from spectraclass.test.texture.util import *
from spectraclass.test.texture.util import load_test_data
from skimage.filters import gabor, gaussian
from scipy import ndimage as ndi
sr2 = math.sqrt( 2.0 )
fig: Figure
block = [2, 1]                  # Set to None to use test data in project

if block is not None: DataManager.initialize("demo1", 'keelin')
dataset_type = "chr"
dsid = "ks"
data_type = "raw"
nGaborAngles = 6
freqs = [ sr2, 1+sr2, 2+sr2, 2*sr2 ]
thetas = list(np.arange(0, np.pi, np.pi / nGaborAngles))
nFeatures = 1
smoothing = 2.0
nBands = 7

t1 = time.time()
fig, axs = plt.subplots( nFeatures+1, nBands )

for iB in range(nBands):
    image: np.ndarray = load_test_data( dataset_type, dsid, data_type, iB, block ).data
    print( f"\nLoaded {data_type} image, band = {iB}, shape = {image.shape}, range = {[ image.min(), image.max() ]}")

    bandwidth = 0.1
    t0 = time.time()
    magnitude_dict = {}
    for iT, theta in enumerate(thetas):
        for iF, freq in enumerate(freqs):
            filt_real, filt_imag = gabor(image, frequency=freq, bandwidth=bandwidth, theta=theta)
            magnitude = get_magnitude([filt_real, filt_imag]).reshape( image.shape )
            magnitude_dict[(iT, iF)] = magnitude.reshape(image.size)
    print( f" Computed Gabor {len(magnitude_dict)} transforms in time {time.time()-t0} sec ")

    fig.suptitle(f'Smoothed Gabor Filter Magnitudes, BW={bandwidth:.3f}', fontsize=12)

    t1 = time.time()
    gabor_mag = []
    for (iT, iF), gmag in magnitude_dict.items():
        theta, freq = thetas[iT], freqs[iF]
        gabor_mag.append( gaussian( gmag, sigma = smoothing*freq ) )
    standardized_data = StandardScaler().fit_transform( np.array( gabor_mag ).reshape( (-1, image.size ) ).T )

    condensed_image = apply_standard_pca( standardized_data, nFeatures )
    print( f" Computed gabor features in time {time.time()-t1} sec, smoothing= {smoothing}, shape = {condensed_image.shape}" )

    plot2( axs, 0, iB, image, f"Band-{iB}" )
    for iF in range(nFeatures):
        gabor_feature = condensed_image[:,iF].reshape( image.shape )
        plot2(axs, iF+1, iB, gabor_feature, f"GF-{iF}")

plt.show()