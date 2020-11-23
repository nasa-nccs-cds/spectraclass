from .base import UMAP
import numpy as np
import xarray as xa
import pandas as pd
from typing import List, Union, Tuple, Optional, Dict, Callable
import cuml, cudf
import time

class gpUMAP(UMAP):

    def __init__( self, *args, **kwargs ):
        UMAP.__init__( self, *args, **kwargs )
        self._mapper = None

    def embed( self, X, y=None, **kwargs ):
        input, mapper = self.getMapper(X, y, **kwargs)
        t0 = time.time()
        self._embedding_ = mapper.transform( input )
        print(f"Completed umap embed in time {time.time() - t0} sec, embedding shape = {self._embedding_.shape}")
        return self

    def getMapper(self, X, y=None, **kwargs ) -> Tuple[cudf.DataFrame,cuml.UMAP]:
        input_data: cudf.DataFrame = self.getDataFrame(X)
        if self._mapper is None:
            t0 = time.time()
            print(f"Computing embedding, input shape = {X.shape}")
            self._mapper = cuml.UMAP(init=self.init, n_neighbors=self.n_neighbors, n_components=self.n_components,
                                n_epochs=self.n_epochs, min_dist=self.min_dist, output_type="numpy")
            self._mapper.fit(input_data)
            print(f"Completed umap fit in time {time.time() - t0} sec")
        return input_data, self._mapper

    def getDataFrame(self, X ) -> cudf.DataFrame:
        if isinstance( X, xa.DataArray) or isinstance( X, np.ndarray ):
            return cudf.DataFrame( { icol: X[:, icol] for icol in range( X.shape[1] ) } )
        elif isinstance( X, cudf.DataFrame ):
            return X
        elif isinstance(X, pd.DataFrame):
            return cudf.DataFrame(X)
        else:
            raise Exception( f"Unsupport input type for gpUMAP: {X.__class__.__name__}")

    def transform(self, X: Union[xa.DataArray,np.ndarray]  ):
        t0 = time.time()
        input_data, mapper = self.getMapper(X)
        result = mapper.transform(input_data)
        print(f"Completed umap transform in time {time.time() - t0} sec, result shape = {result.shape}")
        return result


