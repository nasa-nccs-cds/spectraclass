import traitlets.config as tlc
import os, logging, numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type, Iterable
from traitlets.config.loader import Config

class SCSingletonConfigurable(tlc.Configurable):
    config_instances: List["SCSingletonConfigurable"] = []
    _instance = None
    _instantiated = None

    def __init__(self, *args, **kwargs ):
        super(SCSingletonConfigurable, self).__init__()
        self._contingent_configuration_()
        self.config_instances.append( self )

    def process_event(self, name: str, args: Dict ):
        pass

    def submit_event(self, name: str, args: Dict ):
        for inst in self.config_instances:
            inst.process_event( name, args )

    @classmethod
    def instance(cls, *args, **kwargs):
        if cls._instance is None:
            inst = cls(*args, **kwargs)
            cls._instance = inst
            cls._instantiated = cls
        return cls._instance

    def set_parent_instances( self, current_class: Type["SCSingletonConfigurable"] = None ):  #
        if current_class is None: current_class = self.__class__
        for  base_class in current_class.__bases__:
            if issubclass( base_class, SCSingletonConfigurable ) and (base_class.__name__ != "SCSingletonConfigurable"):
                assert base_class._instance is None, f"Error, {base_class} cannot be instantiated"
                base_class._instance = self
                base_class._instantiated = self.__class__
                self.set_parent_instances( base_class )

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
        from spectraclass.data.base import DataManager, dm
        class_traits = instance.class_traits(config=True)
#        print(f"  ADDING TRAITS [{instance.__class__}]: {class_traits.keys()}")
        for tid, trait in class_traits.items():
            cache = trait.metadata.get('cache',True)
            if cache:
                tval = getattr(instance, tid)
                trait_scope = dm().name   # instance.config_scope()
                trait_instance_values = trait_map.setdefault( trait_scope, {} )
                trait_values = trait_instance_values.setdefault( instance.__class__.__name__, {} )
    #            print( f"    *** add_trait_value[{instance.config_scope()},{instance.__class__.__name__}]: {tid} -> {tval}")
                trait_values[tid] = tval

    @classmethod
    def get_subclass_instances(cls):
        result = set()
        path = [cls]
        while path:
            parent = path.pop()
            for child in parent.__subclasses__():
                if not '.' in str(child):
                    continue
                if child not in result:
                    result.add(child)
                    path.append(child)
        return [ s.instance() for s in result ]
