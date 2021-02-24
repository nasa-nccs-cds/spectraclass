import matplotlib.pyplot as plt
from functools import partial
from spectraclass.test.texture.util import *
from skimage.feature import greycomatrix, greycoprops
from skimage.util import apply_parallel, img_as_ubyte
from skimage.filters import gaussian
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

t0 = time.time()
def glcm_feature( feature: str, patch: np.ndarray ):
    glcm = greycomatrix( patch, glcm_distances, glcm_angles, levels, symmetric=True, normed=True )
    f = greycoprops( glcm, feature )[0, 0]
    return np.full( patch.shape, f )

raw_image: np.ndarray = load_test_data( dataset_type, dsid, data_type, iLayer ).data[:block_size,:block_size]
image: np.ndarray = rescale( raw_image, bin_size )
print( f"Loaded {data_type} image, band = {iLayer}, shape = {image.shape}, range = {[ image.min(), image.max() ]}, in time {time.time()-t0} sec")

t1= time.time()

h = apply_parallel( partial( glcm_feature, 'homogeneity' ), image, chunks=grid_size, depth=overlap, mode='reflect' )
homogeneity = gaussian( h, sigma = grid_size/2,  mode='reflect' )

e = apply_parallel( partial( glcm_feature, 'energy' ), image, chunks=grid_size, depth=overlap, mode='reflect' )
energy = gaussian( e, sigma = grid_size/2,  mode='reflect' )

print( f"Computed glcm_features in time {time.time()-t1} sec")

fig, axs = plt.subplots( 1, 3 )
plot( axs, 0, image,         f"Image" )
plot( axs, 1, homogeneity,   f"GLCM-homogeneity" )
plot( axs, 2, energy,        f"GLCM-energy" )
plt.show()

