from reduction.base import UMAP
import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import pandas as pd
import cuml, cudf
import time

class gpUMAP(UMAP):

    def __init__( self, *args, **kwargs ):
        UMAP.__init__( self, *args, **kwargs )
        self._mapper = None

    def embed( self, X, y=None, **kwargs ):
        input = self.trainMapper(X, y, **kwargs)
        t0 = time.time()
        self._embedding_ = self._mapper.transform( input, knn_graph = kwargs.get( 'nngraph', self.getNNGraph()) )
        print(f"Completed umap embed in time {time.time() - t0} sec, embedding shape = {self._embedding_.shape}")
        return self

    def trainMapper(self, X, y=None, **kwargs) -> Tuple[cudf.DataFrame, cuml.UMAP]:
        from ..graph.manager import ActivationFlowManager, afm, ActivationFlow
        input_data: cudf.DataFrame = self.getDataFrame(X)
        if self._mapper is None:
            t0 = time.time()
            print(f"Computing embedding, input shape = {X.shape}")
            self._mapper = cuml.UMAP(init=self.init, n_neighbors=self.n_neighbors, n_components=self.n_components, n_epochs=self.n_epochs, min_dist=self.min_dist, output_type="numpy")
            self._mapper.fit(input_data, knn_graph = kwargs.get( 'nngraph', self.getNNGraph()) )
            print(f"Completed umap fit in time {time.time() - t0} sec")
        return input_data

    def getDataFrame(self, X ) -> cudf.DataFrame:
        if isinstance( X, xa.DataArray) or isinstance( X, np.ndarray ):
            return cudf.DataFrame( { icol: X[:, icol] for icol in range( X.shape[1] ) } )
        elif isinstance( X, cudf.DataFrame ):
            return X
        elif isinstance(X, pd.DataFrame):
            return cudf.DataFrame(X)
        else:
            raise Exception( f"Unsupported input type for gpUMAP: {X.__class__.__name__}")

    def transform(self, X: Union[xa.DataArray,np.ndarray]  ):
        t0 = time.time()
        input_data = self.trainMapper(X)
        result = self._mapper.transform( input_data, knn_graph = self.getNNGraph() )
        print(f"Completed umap transform in time {time.time() - t0} sec, result shape = {result.shape}")
        return result


