import math, random, numpy as np
from spectraclass.test.texture.util import *
from skimage.feature import greycomatrix, greycoprops
from skimage.util import apply_parallel, img_as_ubyte, pad
from skimage.transform import pyramid_expand
pi4 = math.pi / 4

class GLCM:

    def __init__(self, **kwargs):
        self.grid_size = kwargs.get('grid_size', 3)
        self.overlap = kwargs.get('overlap', 2)
        self.bin_size = kwargs.get('bin_size', 4)
        self.distances = kwargs.get('distances', [1, 2, 3, 4])
        self.angles = kwargs.get('angles', [0, pi4, 2 * pi4, 3 * pi4])

    def _rescale(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - X.min()) / (X.max() - X.min())
        return img_as_ubyte(Xs) // self.bin_size

    def _unpack(self, image: np.ndarray, padding: List[Tuple[int,int]], offset: int ) -> np.ndarray:
        fdata = image[0::self.grid_size, offset::self.grid_size].reshape([s // self.grid_size for s in image.shape])
        upscaled = pyramid_expand(fdata, upscale=self.grid_size, sigma=None, order=1, mode='reflect', cval=0, multichannel=False, preserve_range=False)
        return pad( upscaled, padding, mode='edge' )

    def _glcm_feature(self, patch: np.ndarray):
        levels = 256 // self.bin_size
        if patch.size == 1: return np.zeros_like( patch, dtype=np.float )
        glcm = greycomatrix( patch, self.distances, self.angles, levels, symmetric=True, normed=True )
        h = greycoprops( glcm, 'homogeneity' )[0, 0]
        rv: np.ndarray = np.full( patch.shape, h, dtype=np.float )
        rv[self.overlap, self.overlap] = greycoprops( glcm, 'energy' )[0, 0]
        return rv

    def compute_features(self, input_data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:  # input_data: dims = [ band, y, x ]
        block_size = [ (s // self.grid_size) * self.grid_size for s in input_data.shape ]
        padding = [ ( 0, input_data.shape[i] - block_size[i]) for i in (0,1) ]
        raw_image: np.ndarray = input_data[ :block_size[0], :block_size[1] ]
        image: np.ndarray = self._rescale(raw_image)
        features = apply_parallel( self._glcm_feature, image, chunks=self.grid_size, depth=self.overlap, mode='reflect' )
        energy: np.ndarray = self._unpack(features, padding, 0)
        homogeneity: np.ndarray = self._unpack(features, padding, 1)
        return ( homogeneity, energy )