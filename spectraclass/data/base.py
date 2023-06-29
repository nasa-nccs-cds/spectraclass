import warnings
warnings.simplefilter("ignore", FutureWarning)
import numpy as np
import panel as pn
import shutil
import ipywidgets as ipw
from typing import List, Union, Tuple, Optional, Dict, Type
import os, warnings
from panel.layout import Panel
from enum import Enum
import ipywidgets as ip
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.application.controller import SpectraclassController
import xarray as xa
import traitlets as tl
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module
from spectraclass.model.base import SCSingletonConfigurable
from traitlets.config.loader import load_pyconfig_files
from .modes import ModeDataManager
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from traitlets.config.loader import Config, PyFileConfigLoader
from spectraclass.data.modes import BlockSelectMode
import threading, time, logging, sys

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
        for ( _, module_name, _ ) in iter_modules( [ str(package_dir) ] ):
            if module_name == "modes":
                # import the module and iterate through its attributes
                module = import_module(f"spectraclass.data.{mode_class}.{module_name}")
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if isclass(attribute) and issubclass(attribute, ModeDataManager):
                        if attribute.MODE is not None:
                            # Add the class to this texture's variables
                            globals()[attribute_name] = attribute
                            DataManager.register_mode(attribute)

class DataManager(SCSingletonConfigurable):
    proc_type = tl.Unicode('cpu').tag(config=True)
    labels_dset = tl.Unicode('labels').tag(config=True)
    _mode_data_managers_: Dict = {}
    refresh_data = tl.Bool(False).tag(config=True, sync=True)
    preprocess = tl.Bool(False).tag(config=True, sync=True)

    def __init__(self):
        self.config_files = []
        self.name = None
        self._mode_data_manager_: ModeDataManager = None
        self._project_data: xa.Dataset = None
        super(DataManager, self).__init__()
        self._wGui = None
        self._lock = threading.Lock()

    def _contingent_configuration_(self):
        pass

    def getXarray( self, id: str, xcoords: Dict, xdims: Dict, **kwargs ) -> xa.DataArray:
        np_data: np.ndarray = self.getInputFileData( id )
        dims, coords = [], {}
        for iD, iS in enumerate( np_data.shape ):
            dim = xdims.get(iS,f"dim{iD}")
            dims.append( dim )
            if dim in xcoords:
                coords[dim] = xcoords[dim]
            else:
                xdims[iS] = dim
                xcoords[dim] = np.arange(iS)
        result =  xa.DataArray( np_data, dims=dims, coords=coords, name=id, attrs={**kwargs, 'name': id} )
        lgm().log(f"Creating Xarray[{id}], shape={result.shape}, dims={dims}, dtype={result.dtype}, data strides={np_data.strides}")
        return result

    def clear_project_cache(self):
        self._project_data = None

    def getClassMap(self)-> Optional[xa.DataArray]:
        return self.modal.getClassMap()

    @property
    def sysconfig(self) -> Config:
        return self.config

    @classmethod
    def initialize(cls, name: str, mode: str ):
#        try: tf.enable_eager_execution()
#        except: pass
        name = name.lower()
        mode = mode.lower()
        dataManager = cls.instance()
        if dataManager.name is None:
            lgm().init_logging(name, mode)
            dataManager._configure_( name, mode )
            if mode.lower() not in cls._mode_data_managers_: raise Exception( f"Mode {mode} is not defined, available modes = {cls._mode_data_managers_.keys()}")
            mdm = cls._mode_data_managers_[ mode ].instance()
            dataManager._mode_data_manager_ = mdm
            lgm().log("Logging configured")
        return dataManager

    def hasMetadata(self):
        return os.path.isfile( self.metadata_file )

    def preprocess_data(self):
        if not self.modal.hasBlockData() or not self.hasMetadata():
            self.prepare_inputs( )
            self.save_config()

    def app(self) -> SpectraclassController:
        if self.preprocess: self.prepare_inputs()
        return self.modal.application.instance()

    @property
    def defaults_dir(self):
        sc_dir = os.path.dirname( os.path.dirname( os.path.realpath("__file__") ) )
        return os.path.join( sc_dir, "defaults" )

    def _configure_(self, name: str, mode: str ):
        self.name = name
        self.config_files.append( [ self.defaults_dir, "config.py" ] )
        user_config_file = [ self.config_dir(  mode ), f"{name}.py" ]
        ucfile = os.path.join(*user_config_file)
        if not os.path.isfile( ucfile ):
            def_ucfile = os.path.join( os.path.dirname(os.getcwd()), "defaults", mode, f"{name}.py" )
            if os.path.isfile( def_ucfile ):
                lgm().log( f"Installing config file from {def_ucfile} to {ucfile}", print=True )
                shutil.copyfile( def_ucfile, ucfile )
            else:
                lgm().log(f"Default config file not found: {def_ucfile}", print=True)
        self.config_files.append( user_config_file )
        for ( cfg_dir, fname) in self.config_files:
            cfg_file = os.path.join( cfg_dir, fname )
            if os.path.isfile(cfg_file):
                lgm().log( f"Using config file: '{cfg_file}'", print=True )
                loader = PyFileConfigLoader( fname, path=cfg_dir )
                self.update_config( loader.load_config() )
            else:
                lgm().log(f" ---> Config file not found: {cfg_file}", print=True)

    def getCurrentConfig(self):
        cfg = Config()
        config_dict = { dm().name: cfg }
        for ( cfg_dir, fname ) in self.config_files:
            if os.path.isfile( os.path.join( cfg_dir, fname ) ):
                loader = PyFileConfigLoader( fname, path=cfg_dir )
                cfg.merge( loader.load_config() )
        return config_dict

    @property
    def mask_file(self ) -> str:
        output_dir = os.path.join( self.cache_dir, "masks")
        os.makedirs( output_dir, exist_ok=True)
        return os.path.join( output_dir, f"{self.dsid()}-masks.nc" )

    @property
    def labels_file(self ) -> str:
        output_dir = os.path.join( self.cache_dir, "labels")
        os.makedirs( output_dir, exist_ok=True)
        return os.path.join( output_dir, f"{self.dsid()}-{self.labels_dset}.nc" )

    @exception_handled
    def save_config( self, overwrite = False ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.reduction.embedding import ReductionManager, rm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        afm(), lm(), pcm(), mm(), rm()
        conf_dict = self.generate_config_file()
        for scope, trait_classes in conf_dict.items():
            cfg_file = os.path.realpath( self.config_file( scope, self.mode ) )
            if overwrite or not os.path.exists(cfg_file):
                os.makedirs(os.path.dirname(cfg_file), 0o777, exist_ok=True)
                lines = []

                for class_name, trait_map in trait_classes.items():
                    for trait_name, trait_value in trait_map.items():
                        tval_str = f'"{trait_value}"' if isinstance(trait_value, str) else f"{trait_value}"
                        cfg_str = f"c.{class_name}.{trait_name} = {tval_str}\n"
                        lines.append( cfg_str )

                lgm().log(f"Writing config file: {cfg_file}")
                with self._lock:
                    cfile_handle = open(cfg_file, "w")
                    cfile_handle.writelines(lines)
                    cfile_handle.close()
                lgm().log(f"Config file written")

    def generate_config_file(self) -> Dict:
        #        print( f"Generate config file, _classes = {[inst.__class__ for inst in cls.config_instances]}")
        trait_map = self.getCurrentConfig()
        for inst in self.config_instances:
            self.add_trait_values(trait_map, inst)
        return trait_map

    def refresh_all(self):
        for inst in self.config_instances: inst.refresh()
        lgm().log( "Refreshed Configuration")

    @classmethod
    def register_mode(cls, manager_type: Type[ModeDataManager] ):
#        print( f"DataManager registering ModeDataManager[{manager_type.MODE.lower()}]: {manager_type}")
        cls._mode_data_managers_[ manager_type.MODE.lower() ] = manager_type

    @classmethod
    def config_file( cls, name: str, mode:str ) -> str :
        config_dir = os.path.join( os.path.expanduser("~"), ".spectraclass", "config",  mode )
        if not os.path.isdir( config_dir ): os.makedirs( config_dir, mode = 0o777 )
        return os.path.join( config_dir, name + ".py" )

    @classmethod
    def config_dir( cls, mode:str ) -> str :
        config_dir = os.path.join( os.path.expanduser("~"), ".spectraclass", "config",  mode )
        if not os.path.isdir( config_dir ): os.makedirs( config_dir, mode = 0o777 )
        return config_dir

    @property
    def mode(self) -> str:
        return self._mode_data_manager_.MODE

    @property
    def modal(self) -> ModeDataManager:
        return self._mode_data_manager_

    @property
    def cache_dir(self) -> str:
        return os.path.join( self.modal.cache_dir, "spectraclass", self.modal.MODE, self.name )

    @property
    def metadata_file(self) -> str:
        return f"{self.cache_dir}/{self.modal.image_name}.mdata.txt"

    def dsid(self, **kwargs ) -> str:
        return self._mode_data_manager_.dsid( **kwargs )

    def set_dsid(self, dsid: str ) -> str:
        return self._mode_data_manager_.set_dsid( dsid )

    @property
    def project_name(self) -> str:
        return ".".join( [ self.name, self.mode ] )

    @property
    def table_cols(self) -> List:
        return self._mode_data_manager_.metavars

    def gui( self, mode: BlockSelectMode ) -> pn.Tabs():
        from spectraclass.application.controller import SpectraclassController
        if self._wGui is None:
            SpectraclassController.set_spectraclass_theme()
            mode_gui = self._mode_data_manager_.gui(mode)
            self._wGui = pn.Tabs( [ ("blocks",mode_gui) ] ) # , ("images",dm().images_panel()) ] )
        return pn.Column( self._wGui )

    def ipw_gui( self ) -> ip.Tab():
        from spectraclass.application.controller import SpectraclassController
        if self._wGui is None:
            SpectraclassController.set_spectraclass_theme()
            mode_gui = self._mode_data_manager_.gui()
            self._wGui = ipw.Tab()
            self._wGui.children = [ mode_gui, dm().images_panel() ]
            self._wGui.set_title( 0, "blocks" )
            self._wGui.set_title( 1, "images" )
        return self._wGui

    def images_panel(self) -> Panel:
        title = pn.widgets.StaticText(name='ImagesTitle', value='Images')
        return pn.Column( title, dm().modal.file_selector )

    def on_image_change(self, event: Dict):
        dm().modal.on_image_change( event )

    def getInputFileData(self, vname: str = None, **kwargs ) -> np.ndarray:
        return self._mode_data_manager_.getInputFileData( vname, **kwargs )

    def loadCurrentProject(self, caller_id: str = "main", clear = False, **kwargs ) -> Optional[ Dict[str,Union[xa.DataArray,List,Dict]] ]:
        lgm().log( f" DataManager: loadCurrentProject: {caller_id}" )
        if clear: self._project_data = None
        self.prepare_inputs()
        if self._mode_data_manager_ is not None:
            if self._project_data is None:
                self._project_data = self._mode_data_manager_.loadCurrentProject(**kwargs)
                assert self._project_data is not None, "Project initialization failed- check log file for details"
                ns = self._project_data['samples'].size
                lgm().log(f"LOAD TILE[{self.dsid()}]: #samples = {ns} ")
                if ns == 0: ufm().show( "This tile contains no data","warning")
                self.save_config()
            return self._project_data
        else:
            lgm().log("loadCurrentProject: mode_data_manager is None")

    def loadProject(self, dsid: str = None ) -> Optional[ Dict[str,Union[xa.DataArray,List,Dict]] ]:
        if dsid is not None: self.set_dsid( dsid )
        self._project_data = None
        return self._mode_data_manager_.loadCurrentProject()

    @exception_handled
    def prepare_inputs( self, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        print( " ---------------------------- Project Init ---------------------------- ")
        print( f" *block_size = {TileManager.block_size}")
        print( f" *model_dims = {dm().modal.model_dims}")
        print( f" *anomaly = {dm().modal.anomaly}")
        self._mode_data_manager_.prepare_inputs( **kwargs )

    @exception_handled
    def process_block( self, block, has_metadata: bool ) -> xa.Dataset:
        return self._mode_data_manager_.process_block( block, has_metadata )

    @exception_handled
    def getSpectralData( self, **kwargs ) -> Optional[xa.DataArray]:
        return self._mode_data_manager_.getSpectralData( **kwargs )

    @exception_handled
    def getModelData(self, **kwargs) -> Optional[xa.DataArray]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        raster: bool = kwargs.pop( 'raster', False )
        norm: bool = kwargs.pop( 'norm', False )
        block = kwargs.get( 'block', tm().getBlock() )
        point_data = block.model_data
        if norm: point_data = tm().norm( point_data )
        return block.points2raster(  point_data ) if raster else point_data

    def valid_bands(self) -> Optional[List]:
        return self._mode_data_manager_.valid_bands()


    def getSpatialDims(self) -> Dict[str,int]:
        project_dataset: Optional[ Dict[str,Union[xa.DataArray,List,Dict]] ] = self.loadCurrentProject("getModelData")
        raw_data: xa.DataArray = project_dataset['raw']
        return dict( ny = raw_data.shape[1], nx = raw_data.shape[2] )

    def loadMatlabDataFile(self, file_path: str ):
        from scipy.io import loadmat
        data = loadmat( file_path )
        return data

register_modes()