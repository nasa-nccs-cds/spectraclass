import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
import os, time, threading, traceback
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable

def afm() -> "ActivationFlowManager":
    return ActivationFlowManager.instance()

class ActivationFlow():

    def __init__(self, n_neighbors: int, **kwargs ):
        self.nneighbors = n_neighbors
        self.metric = kwargs.get( 'metric', 'euclidean' )
        self.p = kwargs.get( 'p', 2 )
        self._knn_graph = None

    def get_distances(self) -> np.ndarray:
        raise NotImplementedError()

    def get_classes(self) -> np.ndarray:
        raise NotImplementedError()

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        raise NotImplementedError()

    def getGraph(self):
        raise NotImplementedError()

    @classmethod
    def instance(cls, point_data: xa.DataArray, nneighbors: int, **kwargs ) -> "ActivationFlow":
        from spectraclass.data.base import DataManager
        ptype = DataManager.instance().proc_type
        if ptype == "cpu":
            from .cpu import cpActivationFlow
            return cpActivationFlow( point_data, nneighbors, **kwargs )
        elif ptype == "gpu":
            from .gpu import gpActivationFlow
            return  gpActivationFlow( point_data, nneighbors, **kwargs )
        else:
            raise NotImplementedError( f"Error, unimplemented proc_type: {ptype}")

class ActivationFlowManager(SCSingletonConfigurable):
    nneighbors = tl.Int( 5 ).tag(config=True,sync=True)
    metric = tl.Unicode("euclidean").tag(config=True,sync=True)

    def __init__(self):
        super(ActivationFlowManager, self).__init__()
        self.instances = {}
        self.condition = threading.Condition()

    def getActivationFlow( self ) -> Optional["ActivationFlow"]:
        from spectraclass.data.base import DataManager
        point_data: xa.DataArray = DataManager.instance().getModelData()
        result = None
        if point_data is not None:
            dsid = point_data.attrs.get('dsid','global')
            print( f"Get Activation flow for dsid {dsid}")
            self.condition.acquire()
            try:
                result = self.instances.get( dsid, None )
                if result is None:
                    metric_specs = self.metric.split("-")
                    kwargs = dict( metric = metric_specs[0] )
                    kwargs['p'] = int(metric_specs[1]) if len( metric_specs ) > 1 else 2
                    result = ActivationFlow.instance( point_data, self.nneighbors, **kwargs )
                    self.instances[dsid] = result
                self.condition.notifyAll()
            except Exception as err:
                print( f"Error in getting ActivationFlow: {err}")
                traceback.print_exc()
            finally:
                self.condition.release()
        return result

