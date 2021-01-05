import traitlets.config as tlc
import numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type

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
    def add_trait_values( cls, trait_map: Dict, cname: str, instance: "SCSingletonConfigurable" ):
        class_traits = instance.class_traits(config=True)
#        print(f"  ADDING TRAITS [{instance.__class__}]: {class_traits.keys()}")
        for tid, trait in class_traits.items():
            tval = getattr(instance, tid)
            if trait.__class__.__name__ == "Unicode":  tval = f'"{tval}"'
            trait_values = trait_map.setdefault( instance.config_scope(), {} )
#            print( f"    *** add_trait_value[{instance.config_scope()},{pid(instance)}]: {cname+tid} -> {tval}")
            trait_values[cname + tid] = tval

    def generate_config_file( self, trait_map: Dict  ) -> Dict[str,str]:
#        print( f"Generate config file, classes = {[inst.__class__ for inst in cls.config_instances]}")
        for inst in self.config_instances:
            self.add_trait_values( trait_map, f"c.{inst.__class__.__name__}.", inst )
        result: Dict = {}
        for mode, trait_values in trait_map.items():
            lines = ['']
            for name, value in trait_values.items():
                lines.append( f"{name} = {value}")
            result[ mode ] = '\n'.join(lines)
        return result
