from spectraclass.model.base import SCSingletonConfigurable
import traitlets as tl
import math, random, numpy as np
from sklearn.preprocessing import StandardScaler
from spectraclass.test.texture.util import *
from skimage.filters import gabor, gaussian
sr2 = math.sqrt( 2.0 )

class TextureManager(SCSingletonConfigurable):
    bands = tl.List( tl.Int, None ).tag(config=True)

    def __init__(self):
        super(TextureManager, self).__init__()

    def get_gabor_texture( self, input_data: np.ndarray, nFeatures: int = 1 ) -> np.ndarray:     #  input_data:  [ n_features, n_samples ]
        nGaborAngles = 6
        freqs = [sr2, 1 + sr2, 2 + sr2, 2 * sr2]
        thetas = list(np.arange(0, np.pi, np.pi / nGaborAngles))
        smoothing = 2.0

        for iB in range(input_data.shape[0]):
            image_band: np.ndarray = input_data[iB]
            bandwidth = 0.1
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
            standardized_data = StandardScaler().fit_transform(np.array(gabor_mag).reshape((-1, image_band.size)).T)

            condensed_image = apply_standard_pca( standardized_data, nFeatures )
            return condensed_image