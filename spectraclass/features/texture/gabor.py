import math, random, numpy as np
from typing import List, Optional, Dict, Type
from skimage.filters import gabor, gaussian
from .base import TextureHandler
from sklearn.preprocessing import StandardScaler
from spectraclass.test.texture.util import get_magnitude, apply_standard_pca
sr2 = math.sqrt(2.0)

class Gabor(TextureHandler):

    def __init__(self, **kwargs):
        super(Gabor, self).__init__( **kwargs )
        self.nAngles = kwargs.get('nang', 6)
        self.freqs = kwargs.get('freq', [sr2, 1 + sr2, 2 + sr2, 2 * sr2])
        self.smoothing = kwargs.get('smooth', 2.0)
        self.bandwidth = kwargs.get('bw', 0.1)
        self.nFeatures = kwargs.get( 'nfeat', 1 )

    def compute_band_features(self, image_band: np.ndarray) -> List[np.ndarray]:  # input_data: dims = [ y, x ]
        thetas = list(np.arange(0, np.pi, np.pi / self.nAngles))
        magnitude_dict = {}

        for iT, theta in enumerate(thetas):
            for iF, freq in enumerate(self.freqs):
                filt_real, filt_imag = gabor( image_band, frequency=freq, bandwidth=self.bandwidth, theta=theta )
                magnitude = get_magnitude( [filt_real, filt_imag] ).reshape( image_band.shape )
                magnitude_dict[(iT, iF)] = magnitude.reshape( image_band.size )

        gabor_mag = []
        for (iT, iF), gmag in magnitude_dict.items():
            theta, freq = thetas[iT], self.freqs[iF]
            gabor_mag.append( gaussian( gmag, sigma=self.smoothing * freq ) )

        gabor_transforms = StandardScaler().fit_transform( np.array( gabor_mag ).reshape( (-1, image_band.size)).T )
        gabor_features: np.ndarray = apply_standard_pca(gabor_transforms, self.nFeatures)
        return [ gabor_features[:,iF].reshape( image_band.shape ) for iF in range(self.nFeatures)]