import hydra, os, sys
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str, config_path=None):
    sys.tracebacklimit = 100
    if config_path is None: config_path =  "../../conf"
    Configuration.init( config_name, config_path )

class Configuration:
    _instance = None
    _instantiated = None

    def __init__(self, config_name: str, config_path: str ):
        hydra.initialize( version_base=None, config_path=config_path )
        self.cfg: DictConfig = hydra.compose( config_name, return_hydra_config=True )

    @classmethod
    def init(cls, config_name: str, config_path: str ):
        if cls._instance is None:
            inst = cls(config_name,config_path)
            cls._instance = inst
            cls._instantiated = cls

    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance