import pathlib, glob
from functools import partial
import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable
import os, math, pickle, toml

class SettingsManager:

    def __init__( self, **kwargs ):
        self.project_name = None
        self._config: Dict = None
        self.default_settings: Dict = kwargs.get('defaults',{})

    def initProject(self, name: str, scenario: str, default_settings: Dict ):
        self.project_name = name
        self.scenario = scenario
        self._config = self.default_settings = default_settings
        if os.path.isfile( self.settings_file ):
            self._config.update( toml.load( self.settings_file ) )

    @property
    def config(self) -> Dict:
        assert self._config is not None, "Project not yet initialized"
        return self._config

    def iparm(self, key: str ):
        return int( self.config.get(key) )

    def get_dtype(self, result ):
        if isinstance( result, np.ndarray ): return result.dtype
        else: return np.float64 if type( result[0] ) == "float" else None

    @property
    def settings_dir(self) -> str:
        root_dir = os.path.join( os.path.expanduser("~/.spectraclass"), self.project_name )
        os.makedirs( root_dir, exist_ok=True )
        return root_dir

    @property
    def settings_file(self) -> str:
        return os.path.join( self.settings_dir, f'{self.scenario}.toml' )

    def save(self):
        with open(self.settings_file,"w")  as f:
            toml.dump( self.config, f )