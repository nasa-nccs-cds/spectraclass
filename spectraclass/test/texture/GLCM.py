
from spectraclass.test.texture.util import *
from spectraclass.features.texture.glcm import GLCM

distances = [2,4]
features: List[str] = ["homogeneity", "energy"]
band = 3
band_image: np.ndarray = load_test_data( "chr", "ks", "raw", band ).data
t0 = time.time()

glcm = GLCM( distances=distances, features=features )
tex_features: List[np.ndarray] = glcm.compute_band_features( band_image )

print( f"Computed glcm_features in time {time.time()-t0} sec")

fig, axs = plt.subplots( 1, 3 )
plot( axs, 0, band_image, f"Image" )
for iF, fName in enumerate( features ):
    plot( axs, iF+1, tex_features[iF],   f"GLCM-{fName}" )
plt.show()

