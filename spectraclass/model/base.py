import traitlets.config as tlc
import numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type
from traitlets.config.loader import Config

def pid( instance ): return hex(id(instance))[-4:]

class Marker:
    def __init__(self, pids: Union[List[int],np.ndarray], cid: int ):
        self.cid = cid
        pid_array = pids if isinstance( pids, np.ndarray ) else np.array( pids )
        self.pids = np.unique( pid_array )

    def isTransient(self):
        return self.cid == 0

    def isEmpty(self):
        return self.pids.size == 0

    def deletePid( self, pid: int ) -> bool:
        try:
            self.pids = self.pids[ self.pids != pid ]
            return True
        except: return False

    def deletePids( self, dpids: np.ndarray ) -> bool:
        try:
            self.pids = np.setdiff1d( self.pids, dpids )
            return True
        except: return False

class SCSingletonConfigurable(tlc.LoggingConfigurable):
    config_instances: List["SCSingletonConfigurable"] = []
    _instance = None
    _instantiated = None

    def __init__(self, *args, **kwargs ):
        super(SCSingletonConfigurable, self).__init__()
        self._contingent_configuration_()
        self.config_instances.append( self )

    @classmethod
    def instance(cls, *args, **kwargs):
        if cls._instance is None:
            inst = cls(*args, **kwargs)
            cls._instance = inst
            cls._instantiated = cls
        assert cls._instantiated == cls, f"Error, conflicting singleton instantiations: {cls} vs {cls._instantiated}"
        return cls._instance

    def config_scope(self):
        return "global"

    @classmethod
    def initialized(cls):
        return hasattr(cls, "_instance") and cls._instance is not None

    def _contingent_configuration_(self):
        from spectraclass.data.base import DataManager
        try:                dm = DataManager.instance()
        except TypeError:   raise Exception( f"Error, attempt to instantiate '{self.__class__}' before DataManager is initialized" )
        self.update_config(dm.sysconfig)

    def refresh(self): pass

    @classmethod
    def add_trait_values( cls, trait_map: Dict, instance: "SCSingletonConfigurable" ):
        class_traits = instance.class_traits(config=True)
#        print(f"  ADDING TRAITS [{instance.__class__}]: {class_traits.keys()}")
        for tid, trait in class_traits.items():
            tval = getattr(instance, tid)
            trait_instance_values = trait_map.setdefault( instance.config_scope(), {} )
            trait_values = trait_instance_values.setdefault( instance.__class__.__name__, {} )
#            print( f"    *** add_trait_value[{instance.config_scope()},{pid(instance)}]: {cname+tid} -> {tval}")
            trait_values[tid] = tval


