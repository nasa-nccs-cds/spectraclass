import traitlets.config as tlc
from collections import OrderedDict
import numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type
import traitlets as tl

def pid( instance ): return hex(id(instance))[-4:]

class Marker:
    def __init__(self,  pids: np.ndarray, cid: int ):
        self.cid = cid
        self.pids = np.unique( pids )

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

class SCConfigurable:
    config_classes = set()

    def __init__(self, **kwargs ):
        self.config_classes.add( self.__class__ )

    @property
    def config_mode(self):
        return "global"

    def refresh(self): pass

    @classmethod
    def add_trait_values( cls, trait_map: Dict, cname: str, instance: tlc.Configurable ):
        for tid, trait in instance.class_traits(config=True).items():
            tval = getattr(instance, tid)
            if trait.__class__.__name__ == "Unicode":  tval = f'"{tval}"'
            trait_values = trait_map.setdefault(instance.config_mode, {})
            print( f"    *** add_trait_value[{instance.config_mode},{pid(instance)}]: {cname+tid} -> {tval}")
            trait_values[cname + tid] = tval

    @classmethod
    def generate_config_file( cls ) -> Dict[str,str]:
        trait_map: Dict = {}
        print( f"Generate config file, classes = {[clss.__name__ for clss in cls.config_classes]}")
        for clss in cls.config_classes:
            instance: tlc.Configurable = clss.instance()
            cls.add_trait_values( trait_map, f"c.{instance.__class__.__name__}.", instance )
        result: Dict = {}
        for mode, trait_values in trait_map.items():
            lines = ['']
            for name, value in trait_values.items():
                lines.append( f"{name} = {value}")
            result[ mode ] = '\n'.join(lines)
        return result

    @classmethod
    def _classes_inc_parents(cls):
        """Iterate through configurable classes, including configurable parents """
        seen = set()
        for c in cls.config_classes:
            # We want to sort parents before children, so we reverse the MRO
            for parent in reversed(c.mro()):
                if issubclass(parent, tlc.Configurable) and (parent not in seen):
                    seen.add(parent)
                    yield parent

    @classmethod
    def _classes_with_config_traits(cls):
        """ Yields only classes with configurable traits, and their subclasses.  """
        cls_to_config = OrderedDict( (cls, bool(cls.class_own_traits(config=True))) for cls in cls._classes_inc_parents())

        def is_any_parent_included(cls):
            return any(b in cls_to_config and cls_to_config[b] for b in cls.__bases__)

        ## Mark "empty" classes for inclusion if their parents own-traits, and loop until no more classes gets marked.
        while True:
            to_incl_orig = cls_to_config.copy()
            cls_to_config = OrderedDict( (cls, inc_yes or is_any_parent_included(cls)) for cls, inc_yes in cls_to_config.items())
            if cls_to_config == to_incl_orig:
                break
        for cl, inc_yes in cls_to_config.items():
            if inc_yes:
                yield cl


class AstroModeConfigurable(SCConfigurable):
    _class_instances = {}

    def __init__( self, mode: str ):
        SCConfigurable.__init__(self)
        self._mode = mode
        self._class_instances[mode] = self

    @classmethod
    def instance(cls):
        from spectraclass.data.manager import DataManager
        return cls._class_instances[ DataManager.instance().mode ]

    @property
    def config_mode(self):
        return self._mode