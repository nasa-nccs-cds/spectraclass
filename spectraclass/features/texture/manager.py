from spectraclass.model.base import SCSingletonConfigurable
import traitlets as tl
import math, random, numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.filters import gabor
from spectraclass.test.texture.util import *
from skimage.feature import greycomatrix, greycoprops
from skimage.util import apply_parallel, img_as_ubyte
from skimage.filters import gaussian
from skimage.transform import pyramid_expand
import math

sr2 = math.sqrt(2.0)

def texm(): return TextureManager.instance()

class TextureManager(SCSingletonConfigurable):
    bands = tl.List(tl.Int, None).tag(config=True)

    def __init__(self):
        super(TextureManager, self).__init__()

    def gabor_features(self, input_data: np.ndarray, **kwargs) -> np.ndarray:  # input_data: dims = [ band, y, x ]
        nGaborAngles = kwargs.get('nang', 6)
        freqs = kwargs.get('freq', [sr2, 1 + sr2, 2 + sr2, 2 * sr2])
        thetas = list(np.arange(0, np.pi, np.pi / nGaborAngles))
        smoothing = kwargs.get('smooth', 2.0)
        bandwidth = kwargs.get('bw', 0.1)

        gabor_features = []
        for iB in range(input_data.shape[0]):
            if (len(self.bands) == 0) or iB in self.bands:
                image_band: np.ndarray = input_data[iB]
                t0 = time.time()
                magnitude_dict = {}
                for iT, theta in enumerate(thetas):
                    for iF, freq in enumerate(freqs):
                        filt_real, filt_imag = gabor(image_band, frequency=freq, bandwidth=bandwidth, theta=theta)
                        magnitude = get_magnitude([filt_real, filt_imag]).reshape(image_band.shape)
                        magnitude_dict[(iT, iF)] = magnitude.reshape(image_band.size)
                print(f" Computed Gabor {len(magnitude_dict)} transforms in time {time.time() - t0} sec ")

                t1 = time.time()
                gabor_mag = []
                for (iT, iF), gmag in magnitude_dict.items():
                    theta, freq = thetas[iT], freqs[iF]
                    gabor_mag.append(gaussian(gmag, sigma=smoothing * freq))
                gabor_transforms = StandardScaler().fit_transform(np.array(gabor_mag).reshape((-1, image_band.size)).T)
                gabor_feature = apply_standard_pca(gabor_transforms, 1)
                gabor_features.append(gabor_feature)
                print(f" Computed Gabor feature[{iB}]  in time {time.time() - t1} sec, shape = {gabor_feature.shape} ")
        return np.array(gabor_features).squeeze().reshape(input_data.shape)