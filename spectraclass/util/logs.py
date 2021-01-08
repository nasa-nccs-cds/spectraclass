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
import threading, time, logging, sys, traceback

def lm() -> "LogManager":
    return LogManager.instance()

class LogManager(SCSingletonConfigurable):

    def __init__(self):
        super(LogManager, self).__init__()
        self._wGui = None
        self._level = logging.INFO
        self._log_file = None

    def setLevel(self, level ):
        self._level = level

    def init_logging(self, name: str, mode: str ):
        log_dir = os.path.join( os.path.expanduser("~"), ".spectraclass", "logging", mode )
        os.makedirs( log_dir, exist_ok=True )
        self._log_file = open( f'{log_dir}/{name}.log', 'w' )

    def log( self,  msg, **kwargs ):
        self._log_file.write( msg + "\n" )
        self._log_file.flush()

    def debug(self, msg, **kwargs ):
        if self._level == logging.DEBUG:
            self.log( msg,  **kwargs )

    def exception(self,  msg, **kwargs ):
        self._log_file.write( msg + "\n" )
        self._log_file.write( traceback.format_exc() )
        self._log_file.flush()

    def trace(self,  msg, **kwargs ):
        self._log_file.write( msg + "\n" )
        self._log_file.write( " ".join( traceback.format_stack() ) )
        self._log_file.flush()