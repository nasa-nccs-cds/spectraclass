import matplotlib.pyplot as plt
from functools import partial
from spectraclass.test.texture.util import *
from skimage.feature import greycomatrix, greycoprops
from skimage.util import apply_parallel, img_as_ubyte
from skimage.filters import gaussian
from skimage.transform import pyramid_expand
import math
pi4 = math.pi/4

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iLayer = 0
grid_size = 3
overlap = 2
block_size = (500//grid_size)*grid_size
bin_size = 8

glcm_distances = [ 1, 2, 3, 4 ]
glcm_angles = [ 0, pi4, 2*pi4, 3*pi4 ]
levels = 256 // bin_size

def rescale( X: np.ndarray, bin_size = 1 ) -> np.ndarray:
    Xs = (X - X.min()) / (X.max() - X.min())
    return img_as_ubyte( Xs ) // bin_size

def unpack( image: np.ndarray, grid_size: int, offset: int ):
    fdata = image[0::grid_size,offset::grid_size].reshape( [ s//grid_size for s in image.shape ] )
    return pyramid_expand(fdata, upscale=grid_size, sigma=None, order=1, mode='reflect', cval=0, multichannel=False, preserve_range=False)

t0 = time.time()
def glcm_feature( patch: np.ndarray ):
    if patch.size == 1: return np.zeros_like( patch, dtype = np.float )
    glcm = greycomatrix( patch, glcm_distances, glcm_angles, levels, symmetric=True, normed=True )
    h = greycoprops( glcm, 'homogeneity' )[0, 0]
    rv: np.ndarray = np.full( patch.shape, h, dtype = np.float )
    rv[overlap,overlap] = greycoprops(glcm, 'energy')[0, 0]
    return rv

raw_image: np.ndarray = load_test_data( dataset_type, dsid, data_type, iLayer ).data[:block_size,:block_size]
image: np.ndarray = rescale( raw_image, bin_size )
print( f"Loaded {data_type} image, band = {iLayer}, shape = {image.shape}, range = {[ image.min(), image.max() ]}, in time {time.time()-t0} sec")

t1= time.time()

features = apply_parallel( glcm_feature, image, chunks=grid_size, depth=overlap, mode='reflect' )

energy = unpack( features, grid_size, 0 )
homogeneity = unpack( features, grid_size, 1 )

print( f"Computed glcm_features in time {time.time()-t1} sec")

fig, axs = plt.subplots( 1, 3 )
plot( axs, 0, image,         f"Image" )
plot( axs, 1, homogeneity,   f"GLCM-homogeneity" )
plot( axs, 2, energy,        f"GLCM-energy" )
plt.show()

