import numpy as np
import numpy.ma as ma
from .base import ActivationFlow
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
import os, time, threading, traceback
import traitlets.config as tlc
import traitlets as tl
from spectraclass.model.base import AstroConfigurable

class ActivationFlowManager(tlc.SingletonConfigurable, AstroConfigurable):
    nneighbors = tl.Int( 15 ).tag(config=True,sync=True)

    def __init__(self):
        super(ActivationFlowManager, self).__init__()
        self.instances = {}
        self.condition = threading.Condition()

    def __getitem__( self, dsid ):
        return self.instances.get( dsid )

    def clear(self):
        for instance in self.instances.values():
            instance.clear()

    def getActivationFlow( self, point_data: xa.DataArray, **kwargs ) -> Optional["ActivationFlow"]:
        if point_data is None: return None
        dsid = point_data.attrs.get('dsid','global')
        print( f"Get Activation flow for dsid {dsid}")
        self.condition.acquire()
        try:
            result = self.instances.get( dsid, None )
            if result is None:
                result = ActivationFlow.instance( point_data, self.nneighbors, **kwargs )
                self.instances[dsid] = result
            self.condition.notifyAll()
        finally:
            self.condition.release()
        return result

