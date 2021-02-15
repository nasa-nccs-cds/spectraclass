import skimage.feature.texture as sktex
from spectraclass.data.base import DataManager
from spectraclass.application.controller import app
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.tile.tile import Block
from sklearn.decomposition import PCA
import numpy as np
import xarray as xa
ndims = 4

def pca_reduction( train_input: np.ndarray,  ndim: int ) -> np.ndarray:
    pca: PCA = PCA(n_components=ndim)
    normed_train_input = (train_input - train_input.mean(axis=1,keepdims=True)) / train_input.std(axis=1,keepdims=True)
    pca.fit(normed_train_input)
    print(pca.explained_variance_ratio_ * 100)
    results = pca.transform(train_input)
    return results


dm: DataManager = DataManager.initialize("demo1",'keelin')
block: Block = tm().getBlock()
features, coords = block.getPointData()
print( features.shape )

pca_reduction( features.data, ndims )



# z = sktex.greycomatrix( )
# sktex.local_binary_pattern()
