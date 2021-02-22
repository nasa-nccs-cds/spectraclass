import math, random, numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy
from skimage.morphology import disk
from spectraclass.test.texture.util import *
from spectraclass.test.texture.util import load_test_data
from skimage.filters import gabor, gaussian
from scipy import ndimage as ndi
sr2 = math.sqrt( 2.0 )
fig: Figure

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iLayer = 0
nGaborAngles = 6
freqs = [ sr2, 1+sr2, 2+sr2, 2*sr2 ]
thetas = list(np.arange(0, np.pi, np.pi / nGaborAngles))
smoothing = 1.0
nFeatures = 2
reduction_method = "pca"

t1 = time.time()
fig, axs = plt.subplots( 1, nFeatures + 1 )
image: np.ndarray = load_test_data( dataset_type, dsid, data_type, iLayer ).data
print( f"Loaded {data_type} image, band = {iLayer}, shape = {image.shape}, range = {[ image.min(), image.max() ]}")

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

if reduction_method == "pca":
    condensed_image = apply_standard_pca( standardized_data, nFeatures )
elif reduction_method == "ae":
    condensed_image, reproduction = autoencoder_reduction( standardized_data, nFeatures )
else: raise Exception( f"Unknown reduction_method: {reduction_method}")
print( f" Computed gabor features in time {time.time()-t1} sec, smoothing= {smoothing}, shape = {condensed_image.shape}" )

plot( axs, 0, image, f"Input" )
for iF in range(nFeatures):
    gabor_feature = condensed_image[:,iF].reshape( image.shape )
    plot(axs, iF+1, gabor_feature, f"GF-{iF}")
plt.show()