import numpy as np
from typing import List, Optional, Dict, Tuple, Union
import ipywidgets as ipw
import os, glob, sys
import netCDF4 as nc
import ipywidgets as ip
from os import path
from collections import OrderedDict
from pathlib import Path
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import xarray as xa
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable

def invert( X: np.ndarray ) -> np.ndarray:
    return X.max() - X

class ModeDataManager(SCSingletonConfigurable):
    from spectraclass.application.controller import SpectraclassController

    MODE = None
    METAVARS = None
    INPUTS = None
    VALID_BANDS = None
    application: SpectraclassController = None

    image_names = tl.List( default_value=["NONE"] ).tag(config=True ,sync=True)
    dset_name = tl.Unicode("").tag(config=True)
    cache_dir = tl.Unicode( path.expanduser("~/Development/Cache")).tag(config=True)
    data_dir = tl.Unicode( path.expanduser("~/Development/Data")).tag(config=True)
    class_file = tl.Unicode("NONE").tag(config=True, sync=True)

    model_dims = tl.Int(32).tag(config=True, sync=True)
    subsample_index = tl.Int(1).tag(config=True, sync=True)
    reduce_method = tl.Unicode("Autoencoder").tag(config=True, sync=True)
    reduce_nepochs = tl.Int(5).tag(config=True, sync=True)
    reduce_sparsity = tl.Float( 0.0 ).tag(config=True,sync=True)

    def __init__(self, ):
        super(ModeDataManager,self).__init__()
        assert self.MODE, f"Attempt to instantiate intermediate SingletonConfigurable class: {self.__class__}"
        self.datasets = {}
        self._model_dims_selector: ip.SelectionSlider = None
        self._samples_axis = None
        self._subsample_selector: ip.SelectionSlider = None
        self._progress: ip.FloatProgress = None
        self._dset_selection: ip.Select = None
        self._dataset_prefix: str = ""
        self._file_selector = None
        self._active_image = 0

    def set_current_image(self, image_index: int ):
        lgm().log( f"Setting active_image[{self._active_image}]: {self.image_name}")
        self._active_image = image_index

    @property
    def num_images(self):
        return len( self.image_names )

    @property
    def image_index(self) -> int:
        return self._active_image

    @property
    def image_name(self):
        return self.image_names[self._active_image]

    def get_image_name( self, image_index: int ):
        return self.image_names[ image_index ]

    @property
    def file_selector(self):
        if self._file_selector is None:
            lgm().log( f"Creating file_selector, options={self.image_names}, value={self.image_names[0]}")
            self._file_selector =  ip.Select( options=self.image_names, value=self.image_name, layout=ipw.Layout(width='600px') )
            self._file_selector.observe( self.on_image_change, names=['value'] )
        return self._file_selector

    def on_image_change( self, event: Dict ):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.spatial.map import MapManager, mm
        self._active_image = self.file_selector.index
        dm().clear_project_cache()
        mm().update_plots(True)

    @property
    def mode(self):
        if not self.MODE: raise NotImplementedError(f"Mode {self.MODE} has not been implemented")
        return self.MODE.lower()

    def config_scope(self):
        from spectraclass.data.base import DataManager, dm
        return dm().name

    @property
    def metavars(self):
        return self.METAVARS

    def register(self):
        pass

    def valid_bands(self) -> Optional[List]:
        return self.VALID_BANDS

    def getClassMap(self) -> Optional[xa.DataArray]:
        raise NotImplementedError()

    def set_progress(self, pval: float):
        if self._progress is not None:
            self._progress.value = pval

    def update_gui_parameters(self):
        if self._model_dims_selector is not None:
            self.model_dims = self._model_dims_selector.value
            self.subsample_index = self._subsample_selector.value

    def setDatasetId(self,str):
        raise NotImplementedError()

    def dsid(self, **kwargs) -> str:
        raise NotImplementedError()

    def set_dsid(self, dsid: str ) -> str:
        raise NotImplementedError()

    def prepare_inputs(self, **kwargs ) -> Dict[Tuple,int]:
        raise NotImplementedError()

    def update_extent(self):
        raise NotImplementedError()

    def updateDatasetList(self):
        if self._dset_selection is not None:
            self._dataset_prefix, self._dset_selection.options = self.getDatasetList()

    @property
    def selected_dataset(self):
        return self._dataset_prefix + self._dset_selection.value

    @exception_handled
    def select_dataset(self,*args):
        from spectraclass.data.base import DataManager, dm
        ufm().show( "Loading new data block")
        lgm().log( f"Loading dataset '{self.selected_dataset}', current dataset = '{dm().dsid()}', mdmgr id = {id(self)}")
        dm().loadProject( self.selected_dataset )
        ufm().clear()
        dm().refresh_all()

    def getSelectionPanel(self) -> ip.HBox:
        from spectraclass.data.base import DataManager, dm
        self._dataset_prefix, dsets = self.getDatasetList()
        self._dset_selection: ip.Select = ip.Select(options=dsets, description='Datasets:', disabled=False, layout=ip.Layout(width="900px"))
        if len(dsets) > 0: self._dset_selection.value = dm().dsid()[ len(self._dataset_prefix): ]
        load: ip.Button = ip.Button(description="Load", border='1px solid dimgrey')
        load.on_click(self.select_dataset)
        filePanel: ip.HBox = ip.HBox([self._dset_selection, load], layout=ip.Layout(width="100%", height="100%"), border='2px solid firebrick')
        return filePanel

    def getConfigPanel(self):
        from spectraclass.reduction.embedding import ReductionManager
        rm = ReductionManager.instance()

        nepochs_selector: ip.IntSlider = ip.IntSlider(min=50, max=500, description='UMAP nepochs:', value=rm.nepochs,
                                                      continuous_update=False, layout=ip.Layout(width="auto"))
        alpha_selector: ip.FloatSlider = ip.FloatSlider(min=0.1, max=0.8, step=0.01, description='UMAP alpha:',
                                                        value=rm.alpha, readout_format=".2f", continuous_update=False,
                                                        layout=ip.Layout(width="auto"))
        init_selector: ip.Select = ip.Select(options=["random", "spectral", "autoencoder"],
                                             description='UMAP init method:', value="autoencoder",
                                             layout=ip.Layout(width="auto"))

        def apply_handler(*args):
            rm.nepochs = nepochs_selector.value
            rm.alpha = alpha_selector.value
            rm.init = init_selector.value

        apply: ip.Button = ip.Button(description="Apply", layout=ip.Layout(flex='1 1 auto'), border='1px solid dimgrey')
        apply.on_click(apply_handler)

        configPanel: ip.VBox = ip.VBox([nepochs_selector, alpha_selector, init_selector, apply],
                                       layout=ip.Layout(width="100%", height="100%"), border='2px solid firebrick')
        return configPanel

    def getCreationPanel(self) -> ip.VBox:
        load: ip.Button = ip.Button(description="Create", layout=ip.Layout(flex='1 1 auto'), border='1px solid dimgrey')
        self._model_dims_selector: ip.SelectionSlider = ip.SelectionSlider(options=range(3, 50),
                                                                           description='Model Dimension:',
                                                                           value=self.model_dims,
                                                                           layout=ip.Layout(width="auto"),
                                                                           continuous_update=True,
                                                                           orientation='horizontal', readout=True,
                                                                           disabled=False)

        self._subsample_selector: ip.SelectionSlider = ip.SelectionSlider(options=range(1, 101),
                                                                          description='Subsample:',
                                                                          value=self.subsample_index,
                                                                          layout=ip.Layout(width="auto"),
                                                                          continuous_update=True,
                                                                          orientation='horizontal', readout=True,
                                                                          disabled=False)

        load.on_click(self.prepare_inputs)
        self._progress = ip.FloatProgress(value=0.0, min=0, max=1.0, step=0.01, description='Progress:',
                                          bar_style='info', orientation='horizontal', layout=ip.Layout(flex='1 1 auto'))
        button_hbox: ip.HBox = ip.HBox([load, self._progress], layout=ip.Layout(width="100%", height="auto"))
        creationPanel: ip.VBox = ip.VBox([self._model_dims_selector, self._subsample_selector, button_hbox],
                                         layout=ip.Layout(width="100%", height="100%"), border='2px solid firebrick')
        return creationPanel

    def gui( self, **kwargs ):
        return self.getSelectionPanel()

    def getInputFileData( self, vname: str = None, **kwargs ) -> np.ndarray:
        raise NotImplementedError()

    def getSpectralData( self, **kwargs ) -> Optional[xa.DataArray]:
        raise NotImplementedError()

    def getModelData(self, raw_model_data: xa.DataArray, **kwargs) -> Optional[xa.DataArray]:
        return raw_model_data

    def dset_subsample(self, xdataset, **kwargs) -> Dict[str,Union[xa.DataArray,List]]:
        vnames = xdataset.variables.keys()
        dvars = {}
        for vname in vnames:
            result = variable = xdataset[vname]
            if (self.subsample_index > 1) and (len(variable.dims) >= 1) and (variable.dims[0] == 'samples'):
                if str(variable.dtype) in ["string","object"]:
                    result = variable.values.tolist()
                    result = result[::self.subsample_index]
                else:
                    result = variable[::self.subsample_index] if (variable.ndim == 1) else variable[ ::self.subsample_index, :]
                    result.attrs.update(kwargs)
            dvars[vname] = result
            lgm().log(f" -----> VAR {vname}{result.dims}: shape = {result.shape}")
            if vname in [ 'norm', 'embedding' ]:
                dvars['spectra'] = result
                dvars['plot-y'] = result
        return dvars

    @exception_handled
    def loadDataset(self, **kwargs) -> Optional[ Dict[str,Union[xa.DataArray,List,Dict]] ]:
        if self.dsid() not in self.datasets:
            lgm().log(f"Load dataset {self.dsid()}, current datasets = {self.datasets.keys()}")
            xdataset: xa.Dataset = self.loadDataFile(**kwargs)
            if len(xdataset.variables.keys()) == 0:
                lgm().log(f"Warning: Attempt to Load empty dataset {self.dataFile( **kwargs )}", print=True)
                return None
            else:
                lgm().log(f" ---> Opening Dataset {self.dsid()} from file {xdataset.attrs['data_file']}")
                dvars: Dict[str,Union[xa.DataArray,List,Dict]] = self.dset_subsample( xdataset, dsid=self.dsid(), **kwargs )
                attrs = xdataset.attrs.copy()
                raw_data = dvars['samples']
                lgm().log( f" -----> reduction: shape = {dvars['reduction'].shape}, #NULL={np.count_nonzero(np.isnan(dvars['reduction'].values))}")
                lgm().log( f" -----> point_data: shape = {raw_data.shape}, #NULL={np.count_nonzero(np.isnan(raw_data.values))}")
                dvars['plot-x'] = dvars['bands'] if ('bands'in dvars) else dvars['band']
                dvars['plot-mx'] = dvars['model']
                dvars['plot-mx'] = dvars['model']
                attrs['dsid'] = self.dsid()
                attrs['type'] = 'spectra'
                dvars['attrs'] = attrs
                self.datasets[ self.dsid() ] = dvars
        return self.datasets[ self.dsid() ]

    def blockFilePath( self, **kwargs ) -> str:
        ext = kwargs.get('ext','nc')
        return path.join(self.datasetDir, self.dsid(**kwargs) + "." + ext )

    def removeDataset(self):
        for f in os.listdir(self.datasetDir):
            file = os.path.join(self.datasetDir, f)
            if path.isfile( file ):
                os.remove( file )

    def leafletRasterPath( self, **kwargs ) -> str:
        from spectraclass.data.base import DataManager, dm
        return f"files/spectraclass/datasets/{self.MODE}/{dm().name}/{self.dsid(**kwargs)}.tif"

    def dataFile( self, **kwargs ):
        raise NotImplementedError( "Attempt to call virtual method")

    def hasBlockData(self) -> bool:
        return path.isfile( self.dataFile() )

    def loadDataFile( self, **kwargs ) -> Optional[xa.Dataset]:
        dFile = self.dataFile( **kwargs )
        if path.isfile( dFile ):
            dataset: xa.Dataset = xa.open_dataset( dFile, concat_characters=True )
            dataset.attrs['data_file'] = dFile
            vars = [ f"{vid}{var.dims}" for (vid,var) in dataset.variables.items()]
            coords = [f"{cid}{coord.shape}" for (cid, coord) in dataset.coords.items()]
            lgm().log( f"#GID: loadDataFile: {dFile}, coords={coords}, vars={vars}" )
            lgm().log( f"#GID:  --> coords={coords}")
            lgm().log( f"#GID:  --> vars={vars}")
        else:
            ufm().show( f"This file/tile needs to be preprocesed.", "red" )
            raise Exception( f"Missing data file: {dFile}" )
        return dataset

    def filterCommonPrefix(self, paths: List[str])-> Tuple[str,List[str]]:
        letter_groups, longest_pre = zip(*paths), ""
        for letter_group in letter_groups:
            if len(set(letter_group)) > 1: break
            longest_pre += letter_group[0]
        plen = len( longest_pre )
        return longest_pre, [ p[plen:] for p in paths ]

    def getDatasetList( self ) -> Tuple[str,List[str]]:
        dset_glob = path.expanduser(f"{self.datasetDir}/*.nc")
        lgm().log(f"  Listing datasets from glob: '{dset_glob}' ")
        files = list(filter( path.isfile, glob.glob(dset_glob)))
        files.sort(key=lambda x: path.getmtime(x), reverse=True)
        filenames = [Path(f).stem for f in files]
        return self.filterCommonPrefix( filenames )

    def loadCurrentProject(self) -> Optional[ Dict[str,Union[xa.DataArray,List,Dict]] ]:
        return self.loadDataset( )

    @property
    def datasetDir(self):
        from spectraclass.data.base import DataManager, dm
        dsdir = path.join( self.cache_dir, "spectraclass", dm().mode )
        os.makedirs(dsdir, 0o777, exist_ok=True)
        return dsdir




