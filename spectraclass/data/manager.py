import numpy as np
from typing import List, Union, Tuple, Optional, Dict
import os, math, pickle, glob
import ipywidgets as ip
from functools import partial
from collections import OrderedDict
from spectraclass.reduction.embedding import ReductionManager
from pathlib import Path
import xarray as xa
import traitlets as tl
import traitlets.config as tlc
from spectraclass.model.base import AstroConfigurable, AstroModeConfigurable

class DataManager(tlc.SingletonConfigurable, AstroConfigurable):
    dataset = tl.Unicode("NONE").tag(config=True,sync=True)
    mode_index = tl.Int(0).tag(config=True,sync=True)
    proc_type = tl.Unicode('cpu').tag(config=True)
    MODES = [ "swift", "tess" ]
    METAVARS = dict(swift=["target_names", "obsids"], tess=['tics', "camera", "chip", "dec", 'ra', 'tmag'])
    name = tl.Unicode('spectraclass').tag(config=True)

    def __init__(self):
        super(DataManager, self).__init__()
        self._wModeTabs: ip.Tab = None
        self._init_managers()

    def _init_managers(self):
        self._mode_data_managers = {}
        for iTab, mode in enumerate(self.MODES):
            self._mode_data_managers[iTab] = ModeDataManager(self, mode)

    def config_file(self, config_mode=None) -> str :
        if config_mode is None: config_mode = self.mode
        return os.path.join( os.path.expanduser("~"), "." + self.name, config_mode + ".py" )

    @property
    def mode(self) -> str:
        return self.MODES[ self.mode_index ]

    @property
    def config_mode(self):
        return "configuration"

    @property
    def mode_data_manager(self) -> "ModeDataManager":
        return self._mode_data_managers[ self.mode_index ]

    @property
    def table_cols(self) -> List:
        return self.METAVARS[ self.mode]

    def select_dataset(self, dset: str ):
        self.dataset = dset
        self.mode_index = self._wModeTabs.selected_index
        print( f"Setting Dataset parameters, dataset = {self.dataset}, mode_index = {self.mode_index}")

    def select_current_mode(self):
        if self._wModeTabs is not None:
            self.mode_index = self._wModeTabs.selected_index

    def gui( self ) -> ip.Tab():
        from spectraclass.gui.application import Spectraclass
        if self._wModeTabs is None:
            Spectraclass.set_astrolab_theme()
            mode_tabs = []
            self._wModeTabs = ip.Tab( layout = ip.Layout( width='auto', height='auto' ) )
            for iTab, mdmgr in self._mode_data_managers.items():
                self._wModeTabs.set_title( iTab, self.MODES[iTab]  )
                mode_tabs.append( mdmgr.gui() )
                print( f"DataManager.gui: add ModeDataManager[{iTab}], mode = {mdmgr.config_mode}, mdmgr id = {id(mdmgr)} ")
            self._wModeTabs.children = mode_tabs
            self._wModeTabs.selected_index = self.mode_index
        return self._wModeTabs

    def getInputFileData(self, input_file_id: str, subsample: int = 1, dims: Tuple[int] = None) -> np.ndarray:
        return self.mode_data_manager.getInputFileData( input_file_id, subsample, dims )

    def loadCurrentProject(self, caller_id: str ) -> xa.Dataset:
        print( f" DataManager: loadCurrentProject: {caller_id}" )
        return self.mode_data_manager.loadCurrentProject()

class ModeDataManager( tlc.Configurable, AstroModeConfigurable ):
    model_dims = tl.Int(16).tag(config=True,sync=True)
    subsample = tl.Int( 5 ).tag(config=True,sync=True)
    reduce_method = tl.Unicode("Autoencoder").tag(config=True,sync=True)
    reduce_nepochs = tl.Int( 2 ).tag(config=True,sync=True)
    cache_dir = tl.Unicode( os.path.expanduser("~/Development/Cache") ).tag(config=True)
    data_dir = tl.Unicode( os.path.expanduser("~/Development/Data") ).tag(config=True)

    def __init__(self, dm: DataManager, mode: str, **kwargs ):
        tlc.Configurable.__init__(self)
        AstroModeConfigurable.__init__( self, mode )
        self.datasets = {}
        self._model_dims_selector: ip.SelectionSlider = None
        self._subsample_selector: ip.SelectionSlider = None
        self._progress = None
        self._dset_selection: ip.Select = None
        self.dm = dm

    @classmethod
    def getXarray( cls, id: str, xcoords: Dict, subsample: int, xdims:OrderedDict, **kwargs ) -> xa.DataArray:
        np_data: np.ndarray = DataManager.instance().getInputFileData( id, subsample, tuple(xdims.keys()) )
        dims, coords = [], {}
        for iS in np_data.shape:
            coord_name = xdims[iS]
            dims.append( coord_name )
            coords[ coord_name ] = xcoords[ coord_name ]
        attrs = { **kwargs, 'name': id }
        return xa.DataArray( np_data, dims=dims, coords=coords, name=id, attrs=attrs )

    def get_input_mdata(self):
        if self.config_mode == "swift":
            return dict(embedding='scaled_specs', directory=["target_names", "obsids"], plot=dict(y="specs", x='spectra_x_axis'))
        elif self.config_mode == "tess":
            return dict(embedding='scaled_lcs', directory=['tics', "camera", "chip", "dec", 'ra', 'tmag'], plot=dict(y="lcs", x='times'))
        else:
            raise Exception( f"Unknown data mode: {self.config_mode}, should be 'tess' or 'swift")

    def set_progress(self, pval: float ):
        if self._progress is not None:
            self._progress.value = pval

    def update_gui_parameters(self):
        if self._model_dims_selector is not None:
            self.model_dims = self._model_dims_selector.value
            self.subsample = self._subsample_selector.value

    def prepare_inputs( self, *args ):
        self.dm.select_current_mode()
        self.update_gui_parameters()
        self.set_progress( 0.02 )
        file_name = f"raw" if self.reduce_method == "None" else f"{self.reduce_method}-{self.model_dims}"
        if self.subsample > 1: file_name = f"{file_name}-ss{self.subsample}"
        output_file = os.path.join( self.datasetDir, file_name + ".nc" )

        input_vars = self.get_input_mdata()
        np_embedding: np.ndarray = self.getInputFileData( input_vars['embedding'], self.subsample )
        dims = np_embedding.shape
        mdata_vars = list(input_vars['directory'])
        xcoords = OrderedDict( samples = np.arange( dims[0] ), bands = np.arange(dims[1]) )
        xdims = OrderedDict( { dims[0]: 'samples', dims[1]: 'bands' } )
        data_vars = dict( embedding = xa.DataArray( np_embedding, dims=xcoords.keys(), coords=xcoords, name=input_vars['embedding'] ) )
        data_vars.update( { vid: self.getXarray( vid, xcoords, self.subsample, xdims ) for vid in mdata_vars } )
        pspec = input_vars['plot']
        data_vars.update( { f'plot-{vid}': self.getXarray( pspec[vid], xcoords, self.subsample, xdims, norm=pspec.get('norm','')) for vid in [ 'x', 'y' ] } )
        self.set_progress( 0.1 )
        if self.reduce_method != "None":
           reduced_spectra = ReductionManager.instance().reduce( data_vars['embedding'], self.reduce_method, self.model_dims, self.reduce_nepochs )
           coords = dict( samples=xcoords['samples'], model=np.arange( self.model_dims ) )
           data_vars['reduction'] =  xa.DataArray( reduced_spectra, dims=['samples','model'], coords=coords )
           self.set_progress( 0.8 )

        dataset = xa.Dataset( data_vars, coords=xcoords, attrs = {'type':'spectra'} )
        dataset.attrs["colnames"] = mdata_vars
        print( f"Writing output to {output_file}" )
        dataset.to_netcdf( output_file, format='NETCDF4', engine='netcdf4' )
        self.updateDatasetList()
        self.set_progress( 1.0 )

    def updateDatasetList(self):
        if self._dset_selection is not None:
            self._dset_selection.options = self.getDatasetList()

    def select_dataset(self, *args ):
        from spectraclass.gui.application import Spectraclass
        self.dm.select_current_mode()
        if self.dm.dataset != self._dset_selection.value:
            print(f"Loading dataset '{self._dset_selection.value}', current dataset = '{self.dm.dataset}', current mode = '{self._mode}', current mode index = {self.dm.mode_index}, mdmgr id = {id(self)}")
            self.dm.dataset = self._dset_selection.value
            self.dm.select_dataset( self._dset_selection.value )
        Spectraclass.instance().refresh_all()

    def getSelectionPanel(self ) -> ip.HBox:
        dsets: List[str] = self.getDatasetList()
        self._dset_selection: ip.Select = ip.Select( options = dsets, description='Datasets:',disabled=False )
        if len( dsets ) > 0: self._dset_selection.value = dsets[0]
        load: ip.Button = ip.Button( description="Load", border= '1px solid dimgrey')
        load.on_click(  self.select_dataset )
        filePanel: ip.HBox = ip.HBox( [self._dset_selection, load ], layout=ip.Layout( width="100%", height="100%" ), border= '2px solid firebrick' )
        return filePanel

    def getConfigPanel(self):
        from spectraclass.reduction.embedding import ReductionManager
        rm = ReductionManager.instance()

        nepochs_selector: ip.IntSlider = ip.IntSlider( min=50, max=500, description='UMAP nepochs:', value=rm.nepochs, continuous_update=False, layout=ip.Layout( width="auto" ) )
        alpha_selector: ip.FloatSlider = ip.FloatSlider( min=0.1, max=0.8, step=0.01, description='UMAP alpha:', value=rm.alpha, readout_format=".2f", continuous_update=False, layout=ip.Layout( width="auto" ) )
        init_selector: ip.Select = ip.Select( options=["random","spectral","autoencoder"], description='UMAP init method:', value="autoencoder",  layout=ip.Layout( width="auto" ) )

        def apply_handler(*args):
            from spectraclass.gui.application import Spectraclass
            rm.nepochs = nepochs_selector.value
            rm.alpha = alpha_selector.value
            rm.init = init_selector.value
            Spectraclass.instance().save_config()
        apply: ip.Button = ip.Button(description="Apply", layout=ip.Layout(flex='1 1 auto'), border='1px solid dimgrey')
        apply.on_click( apply_handler )

        configPanel: ip.VBox = ip.VBox( [ nepochs_selector, alpha_selector, init_selector, apply ], layout=ip.Layout( width="100%", height="100%" ), border= '2px solid firebrick' )
        return configPanel

    def getCreationPanel(self) -> ip.VBox:
        load: ip.Button = ip.Button( description="Create", layout=ip.Layout( flex='1 1 auto' ), border= '1px solid dimgrey' )
        self._model_dims_selector: ip.SelectionSlider = ip.SelectionSlider( options=range(3,50), description='Model Dimension:', value=self.model_dims, layout=ip.Layout( width="auto" ),
                                                   continuous_update=True, orientation='horizontal', readout=True, disabled=False  )

        self._subsample_selector: ip.SelectionSlider = ip.SelectionSlider( options=range(1,101), description='Subsample:', value=self.subsample, layout=ip.Layout( width="auto" ),
                                                   continuous_update=True, orientation='horizontal', readout=True, disabled=False  )

        load.on_click( self.prepare_inputs )
        self._progress = ip.FloatProgress( value=0.0, min=0, max=1.0, step=0.01, description='Progress:', bar_style='info', orientation='horizontal', layout=ip.Layout( flex='1 1 auto' ) )
        button_hbox: ip.HBox = ip.HBox( [ load,self._progress ], layout=ip.Layout( width="100%", height="auto" ) )
        creationPanel: ip.VBox = ip.VBox( [ self._model_dims_selector,self._subsample_selector, button_hbox ], layout=ip.Layout( width="100%", height="100%" ), border= '2px solid firebrick' )
        return creationPanel

    def gui( self, **kwargs ) -> ip.Tab():
        wTab = ip.Tab( layout = ip.Layout( width='auto', height='auto' ) )
        selectPanel = self.getSelectionPanel()
        creationPanel = self.getCreationPanel()
        configPanel = self.getConfigPanel()
        wTab.children = [ creationPanel, selectPanel, configPanel ]
        wTab.set_title( 0, "Create")
        wTab.set_title( 1, "Select")
        wTab.set_title( 2, "Configure" )
        return wTab

    def getInputFileData(self, input_file_id: str, subsample: int = 1, dims: Tuple[int] = None ) -> np.ndarray:
        input_file_path = os.path.expanduser( os.path.join( self.data_dir, self.dm.name, self.config_mode, f"{input_file_id}.pkl") )
        try:
            if os.path.isfile(input_file_path):
                print(f"Reading unstructured {input_file_id} data from file {input_file_path}, dims = {dims}")
                with open(input_file_path, 'rb') as f:
                    result = pickle.load(f)
                    if isinstance( result, np.ndarray ):
                        if dims is not None and (result.shape[0] == dims[1]) and result.ndim == 1: return result
                        return result[::subsample]
                    elif isinstance( result, list ):
#                        if dims is not None and ( len(result) == dims[1] ): return result
                        subsampled = [ result[i] for i in range( 0, len(result), subsample ) ]
                        if isinstance( result[0], np.ndarray ):  return np.vstack( subsampled )
                        else:                                    return np.array( subsampled )
            else:
                print( f"Error, the input path '{input_file_path}' is not a file.")
        except Exception as err:
            print(f" Can't read data[{input_file_id}] file {input_file_path}: {err}")

    def loadDataset( self, dsid: str, *args, **kwargs ) -> xa.Dataset:
        print( f"Load dataset {dsid}, current datasets = {self.datasets.keys()}")
        if dsid is None: return None
        if dsid not in self.datasets:
            data_file = os.path.join( self.datasetDir, dsid + ".nc" )
            dataset: xa.Dataset = xa.open_dataset( data_file )
            print( f" ---> Opened Dataset {dsid} from file {data_file}")
            dataset.attrs['dsid'] = dsid
            dataset.attrs['type'] = 'spectra'
            self.datasets[dsid] = dataset
        return self.datasets[dsid]

    def getDatasetList(self):
        dset_glob = os.path.expanduser(f"{self.datasetDir}/*.nc")
        print( f"  Listing datasets from glob: '{dset_glob}' ")
        files = list(filter(os.path.isfile, glob.glob( dset_glob ) ) )
        files.sort( key=lambda x: os.path.getmtime(x), reverse=True )
        return [ Path(f).stem for f in files ]

    def loadCurrentProject(self) -> xa.Dataset:
        return self.loadDataset( self.dm.dataset )

    @property
    def datasetDir(self):
        dsdir = os.path.join( self.cache_dir, self.dm.name, self.config_mode )
        os.makedirs( dsdir, exist_ok=True )
        return dsdir