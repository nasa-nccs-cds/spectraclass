import matplotlib.pyplot as plt
from spectraclass.test.texture.util import *
from skimage.feature import greycomatrix, greycoprops
from skimage.util import apply_parallel, img_as_ubyte
from skimage.filters import gaussian

dataset_type = "chr"
dsid = "ks"
data_type = "raw"
iLayer = 0
grid_size = 5
overlap = 2

glcm_distances = [5]
glcm_angles = [0]
bin_size = 8
levels = 256 // bin_size

def rescale( X: np.ndarray, bin_size = 1 ) -> np.ndarray:
    Xs = (X - X.min()) / (X.max() - X.min())
    return img_as_ubyte( Xs ) // bin_size

t0 = time.time()
def compute_glcm_features( patch: np.ndarray ):
    glcm = greycomatrix( patch, glcm_distances, glcm_angles, levels, symmetric=True, normed=True )
    d = greycoprops(glcm, 'dissimilarity')[0, 0]
    # c = greycoprops(glcm, 'correlation')[0, 0]
    return np.full( patch.shape, d )

image: np.ndarray = rescale( load_test_data( dataset_type, dsid, data_type, iLayer ).data, bin_size )
(ny,nx) = image.shape
print( f"Loaded {data_type} image, band = {iLayer}, shape = {image.shape}, range = {[ image.min(), image.max() ]}, in time {time.time()-t0} sec")

t1= time.time()
# test = compute_glcm_features( image[0:9,0:9] )
result = apply_parallel( compute_glcm_features, image, chunks=grid_size, depth=overlap, mode='reflect' )
print( f"Computed glcm_features, shape = {result.shape}, range = {[ result.min(), result.max() ]}, levels = {levels}, in time {time.time()-t1} sec")

fig, axs = plt.subplots( 1, 2 )
plot(axs, 0, image, f"Input")
plot(axs, 1, result, f"GLCM-dissimilarity")
plt.show()

