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

fig, axs = plt.subplots( len(thetas), len(freqs) )
image: np.ndarray = load_test_data( dataset_type, dsid, data_type, iLayer ).data
(ny,nx) = image.shape
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
for (iT, iF), gmag in magnitude_dict.items():
    theta, freq = thetas[iT], freqs[iF]
    sigma = smoothing*freq
    smoothed_gmag = gaussian( gmag, sigma = sigma )
    plot2(axs, iT, iF, smoothed_gmag.reshape( image.shape ), f"F={freq:.2f}, T={theta:.2f}")

print( f" Computed smoothed Gabor Transforms in time {time.time()-t1} sec, smoothing= {smoothing}" )
plt.show()