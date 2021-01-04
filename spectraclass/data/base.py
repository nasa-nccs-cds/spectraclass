import numpy as np
from typing import List, Optional, Dict, Type
import os
from enum import Enum
import ipywidgets as ip
import xarray as xa
import traitlets as tl
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module
from spectraclass.model.base import SCSingletonConfigurable
from .modes import ModeDataManager
from traitlets.config.loader import Config

class DataType(Enum):
    Embedding = 1
    Plot = 2
    Image = 3
    Directory = 4

def dm() -> "DataManager":
    return DataManager.instance()

def register_modes():
    for mode_class in ['spatial','unstructured']:
        package_dir = Path( __file__ ).resolve().parent.joinpath(mode_class)
        for ( _, module_name, _ ) in iter_modules([package_dir]):
            if module_name == "modes":
                # import the module and iterate through its attributes
                module = import_module(f"spectraclass.data.{mode_class}.{module_name}")
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if isclass(attribute) and issubclass(attribute, ModeDataManager):
                        if attribute.MODE is not None:
                            # Add the class to this package's variables
                            globals()[attribute_name] = attribute
                            DataManager.register_mode(attribute)

class DataManager(SCSingletonConfigurable):
    proc_type = tl.Unicode('cpu').tag(config=True)
    _mode_data_managers_: Dict[str,Type[ModeDataManager]] = {}

    def __init__(self):
        self._config: Config = None
        self.name = None
        self._mode_data_manager_: ModeDataManager = None
        super(DataManager, self).__init__()
        self._wGui = None
        self.auto_write = False

    def _contingent_configuration_(self):
        pass

    @property
    def sysconfig(self) -> Config:
        if self._config is None: raise TypeError( "DataManager not initialized" )
        return self._config

    @classmethod
    def initialize(cls, name: str, mode: str):
        dataManager = cls.instance()
        dataManager._configure_( name, mode )
        if mode.lower() not in cls._mode_data_managers_: raise Exception( f"Mode {mode} is not defined, available modes = {cls._mode_data_managers_.keys()}")
        dataManager._mode_data_manager_ = cls._mode_data_managers_[ mode.lower() ].instance()
        return dataManager

    def _configure_(self, name: str, mode: str ):
        self.name = name
        cfg_file = self.config_file( name, mode )
        from traitlets.config.loader import load_pyconfig_files
        if os.path.isfile(cfg_file):
            (dir, fname) = os.path.split(cfg_file)
            config_files = ['global.py', fname]
            print(f"Loading config files: {config_files} from dir {dir}")
            self._config = load_pyconfig_files(config_files, dir)
            self.update_config( self._config )
        else:
            print(f"Configuration error: '{cfg_file}' is not a file.")

    def save_config( self, conditional = False ):
        if not conditional or (self.auto_write == True):
            self.auto_write = True
            conf_dict = self.generate_config_file()
            for scope, mode_conf_txt in conf_dict.items():
                cfg_file = os.path.realpath( self.config_file( self.name, scope ) )
                os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
                with open(cfg_file, "w") as cfile_handle:
                    print(f"Writing config file: {cfg_file}")
                    cfile_handle.write(mode_conf_txt)

    def refresh_all(self):
        self.save_config()
        for inst in self.config_instances: inst.refresh()
        print( "Refreshed Configuration")

    @classmethod
    def register_mode(cls, manager_type: Type[ModeDataManager] ):
        print( f"DataManager registering ModeDataManager[{manager_type.MODE.lower()}]: {manager_type}")
        cls._mode_data_managers_[ manager_type.MODE.lower() ] = manager_type

    @classmethod
    def config_file( cls, name: str, scope:str ) -> str :
        config_dir = os.path.join( os.path.expanduser("~"), ".spectraclass", "config",  name )
        os.makedirs( config_dir, mode = 0o777, exist_ok = True )
        return os.path.join( config_dir, scope + ".py" )

    @property
    def mode(self) -> str:
        return self._mode_data_manager_.MODE

    @property
    def modal(self) -> ModeDataManager:
        return self._mode_data_manager_

    @property
    def cache_dir(self) -> str:
        return os.path.join( self.modal.cache_dir, self.name, self.modal.MODE )

    @property
    def dataset(self) -> str:
        return self._mode_data_manager_.dataset

    @property
    def project_name(self) -> str:
        return ".".join( [ self.name, self.mode ] )

    @property
    def config_mode(self):
        return "global"

    @property
    def table_cols(self) -> List:
        return self._mode_data_manager_.metavars

    def gui( self ) -> ip.Tab():
        from spectraclass.gui.unstructured.application import Spectraclass
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

    def valid_bands(self) -> Optional[List]:
        return self._mode_data_manager_.valid_bands()

    def execute_task(self, task: str ):
        return self._mode_data_manager_.execute_task(task)

    def graph_flow(self, niters: int = 1 ):
        return self._mode_data_manager_.spread_selection( niters )

    def distance(self, niters: int = 100 ):
        return self._mode_data_manager_.display_distance( niters )

    def getModelData(self) -> xa.DataArray:
        project_dataset: xa.Dataset = self.loadCurrentProject("getModelData")
        model_data: xa.DataArray = project_dataset['reduction']
        model_data.attrs['dsid'] = project_dataset.attrs['dsid']
        return model_data

    def __delete__(self, instance):
        self.save_config()

register_modes()