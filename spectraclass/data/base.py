import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Type
import os, math, pickle, glob
from enum import Enum
import ipywidgets as ip
import xarray as xa
import traitlets as tl
import traitlets.config as tlc
from spectraclass.model.base import SCConfigurable, AstroModeConfigurable
from .modes import ModeDataManager

class DataType(Enum):
    Embedding = 1
    Plot = 2
    Image = 3
    Directory = 4

class DataManager(tlc.SingletonConfigurable, SCConfigurable):
    proc_type = tl.Unicode('cpu').tag(config=True)
    name = tl.Unicode('spectraclass').tag(config=True)
    _mode_data_managers_: Dict[str,Type[ModeDataManager]] = {}

    @classmethod
    def initialize(cls, name: str, mode: str):
        dataManager = cls.instance()
        dataManager.name = name
        if mode.lower() not in cls._mode_data_managers_: raise Exception( f"Mode {mode} is not defined")
        dataManager._mode_data_manager_ = cls._mode_data_managers_[ mode.lower() ]()
        return dataManager

    @classmethod
    def register_mode(cls, manager_type: Type[ModeDataManager] ):
        cls._mode_data_managers_[ manager_type.MODE.lower() ] = manager_type

    def __init__(self):
        super(DataManager, self).__init__()
        self._mode_data_manager_: ModeDataManager = None
        self._wGui = None

    def config_file(self, config_mode=None) -> str :
        if config_mode is None: config_mode = self.mode
        config_dir = os.path.join( os.path.expanduser("~"), ".spectraclass", "config",  self.name )
        os.makedirs( config_dir, mode = 0o777, exist_ok = True )
        return os.path.join( config_dir, config_mode + ".py" )

    @property
    def mode(self) -> str:
        return self._mode_data_manager_.mode

    @property
    def project_name(self) -> str:
        return ".".join( [ self.name, self.mode ] )

    @property
    def config_mode(self):
        return "configuration"

    @property
    def table_cols(self) -> List:
        return self._mode_data_manager_.metavars

    def gui( self ) -> ip.Tab():
        from spectraclass.gui.application import Spectraclass
        if self._wGui is None:
            Spectraclass.set_spectraclass_theme()
            self._wGui = self._mode_data_manager_.gui()
        return self._wGui

    def getInputFileData(self) -> np.ndarray:
        return self._mode_data_manager_.getInputFileData( )

    def loadCurrentProject(self, caller_id: str ) -> xa.Dataset:
        print( f" DataManager: loadCurrentProject: {caller_id}" )
        return self._mode_data_manager_.loadCurrentProject()

    def prepare_inputs( self, *args, **kwargs ) -> xa.Dataset:
        return self._mode_data_manager_.prepare_inputs( *args, **kwargs )

