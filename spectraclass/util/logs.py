import numpy as np
from typing import List, Optional, Dict, Type
import os
from enum import Enum
from functools import wraps
from time import time
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from spectraclass.model.base import SCSingletonConfigurable
import threading, time, logging, sys, traceback

def lgm() -> "LogManager":
    return LogManager.instance()

def exception_handled(func):
    def wrapper( *args, **kwargs ):
        try:        return func( *args, **kwargs )
        except:     lgm().exception( f" Error in {func}:")
    return wrapper

def log_timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        lgm().log( 'EXEC %r args:[%r, %r] took: %2.4f sec' %  (f.__name__, args, kw, te-ts) )
        return result
    return wrap

class LogManager(SCSingletonConfigurable):

    def __init__(self):
        super(LogManager, self).__init__()
        self._wGui = None
        self._level = logging.INFO
        self._log_file = None
        self.log_dir = None

    @classmethod
    def pid(cls):
        return os.getpid()

    def setLevel(self, level ):
        self._level = level

    def init_logging(self, name: str, mode: str, **kwargs ):
        self.log_dir = os.path.join( os.path.expanduser("~"), ".spectraclass", "logging", mode )
        os.makedirs( self.log_dir, 0o777, exist_ok=True )
        overwrite = kwargs.get( 'overwrite', True )
        lid = "" if overwrite else f"-{os.getpid()}"
        log_file = f'{self.log_dir}/{name}{lid}.log'
        self._log_file = open( log_file, 'w' )
        print( f"Opening log file:  '{log_file}'" )

    def log( self,  msg, **kwargs ):
        if kwargs.get( 'print', False ): print( msg )
        self._log_file.write( msg + "\n" )
        self._log_file.flush()

    def fatal(self, msg: str, status: int = 1 ):
        print( msg )
        self._log_file.write( msg + "\n" )
        self._log_file.flush()
        sys.exit( status )

    def debug(self, msg, **kwargs ):
        if self._level == logging.DEBUG:
            self.log( msg,  **kwargs )

    def exception(self,  msg, **kwargs ):
        self._log_file.write( f"\n{msg}\n{traceback.format_exc()}\n" )
        self._log_file.flush()

    def trace(self,  msg, **kwargs ):
        strace = "\n".join(traceback.format_stack())
        self._log_file.write( f"\n{msg}\n{strace}\n" )
        self._log_file.flush()