import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict

class ActivationFlow(object):
    _instance: "ActivationFlow" = None

    def __init__(self, n_neighbors: int, **kwargs ):
        self.nneighbors = 5 # n_neighbors
        self.reset = True

    def clear(self):
        self.reset = True

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        raise NotImplementedError()

    @classmethod
    def instance(cls, point_data: xa.DataArray, nneighbors: int, **kwargs ) -> "ActivationFlow":
        from spectraclass.data.manager import DataManager
        ptype = DataManager.instance().proc_type
        if cls._instance is None:
            if ptype == "cpu":
                from .cpu import cpActivationFlow
                cls._instance = cpActivationFlow( point_data, nneighbors, **kwargs )
            elif ptype == "gpu":
                from .gpu import gpActivationFlow
                cls._instance =  gpActivationFlow( point_data, nneighbors, **kwargs )
            else:
                print( f"Error, unknown proc_type: {ptype}")
        return cls._instance




