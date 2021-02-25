import matplotlib.pyplot as plt
from spectraclass.test.texture.util import *
from spectraclass.features.texture.glcm import GLCM

distances = [2,4]
band = 3
band_image: np.ndarray = load_test_data( "chr", "ks", "raw", band ).data
t0 = time.time()

glcm = GLCM( distances=distances )
features: Tuple[np.ndarray,np.ndarray] = glcm.compute_band_features(band_image)

print( f"Computed glcm_features in time {time.time()-t0} sec")

fig, axs = plt.subplots( 1, 3 )
plot( axs, 0, band_image,         f"Image" )
plot( axs, 1, features[0],   f"GLCM-homogeneity" )
plot( axs, 2, features[1],        f"GLCM-energy" )
plt.show()

