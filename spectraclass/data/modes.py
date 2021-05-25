import numpy as np
from typing import List, Optional, Dict, Tuple
import os, glob, sys
import ipywidgets as ip
from collections import OrderedDict
from pathlib import Path
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled
import xarray as xa
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable, Marker

def invert( X: np.ndarray ) -> np.ndarray:
    return X.max() - X

class ModeDataManager(SCSingletonConfigurable):
    from spectraclass.application.controller import SpectraclassController

    MODE = None
    METAVARS = None
    INPUTS = None
    VALID_BANDS = None
    application: SpectraclassController = None

    image_name = tl.Unicode("NONE").tag(config=True ,sync=True)
    cache_dir = tl.Unicode(os.path.expanduser("~/Development/Cache")).tag(config=True)
    data_dir = tl.Unicode(os.path.expanduser("~/Development/Data")).tag(config=True)

    model_dims = tl.Int(32).tag(config=True, sync=True)
    subsample = tl.Int(1).tag(config=True, sync=True)
    reduce_method = tl.Unicode("Autoencoder").tag(config=True, sync=True)
    reduce_scope = tl.Unicode("block").tag(config=True, sync=True)
    reduce_nepochs = tl.Int(5).tag(config=True, sync=True)
    reduce_sparsity = tl.Float( 0.0 ).tag(config=True,sync=True)

    def __init__(self, ):
        super(ModeDataManager,self).__init__()
        assert self.MODE, f"Attempt to instantiate intermediate SingletonConfigurable class: {self.__class__}"
        self.datasets = {}
        self._model_dims_selector: ip.SelectionSlider = None
        self._subsample_selector: ip.SelectionSlider = None
        self._progress = None
        self._dset_selection: ip.Select = None
        self._dataset_prefix = ""

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

    @classmethod
    def getXarray(cls, id: str, xcoords: Dict, subsample: int, xdims: OrderedDict, **kwargs) -> xa.DataArray:
        from .base import DataManager
        np_data: np.ndarray = DataManager.instance().getInputFileData(id)
        dims, coords = [], {}
        for iS in np_data.shape:
            coord_name = xdims[iS]
            dims.append(coord_name)
            coords[coord_name] = xcoords[coord_name]
        attrs = {**kwargs, 'name': id}
        return xa.DataArray(np_data, dims=dims, coords=coords, name=id, attrs=attrs)

    def set_progress(self, pval: float):
        if self._progress is not None:
            self._progress.value = pval

    def update_gui_parameters(self):
        if self._model_dims_selector is not None:
            self.model_dims = self._model_dims_selector.value
            self.subsample = self._subsample_selector.value

    def setDatasetId(self,str):
        raise NotImplementedError()

    def dsid(self, **kwargs) -> str:
        raise NotImplementedError()

    def prepare_inputs(self, *args, **kwargs):
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
        if dm().dsid() != self.selected_dataset:
            ufm().show( "Loading new data block")
            lgm().log( f"Loading dataset '{self.selected_dataset}', current dataset = '{dm().dsid()}', mdmgr id = {id(self)}")
            dm().loadProject( self.selected_dataset )
            ufm().clear()
        dm().refresh_all()

    def getSelectionPanel(self) -> ip.HBox:
        self._dataset_prefix, dsets = self.getDatasetList()
        self._dset_selection: ip.Select = ip.Select(options=dsets, description='Datasets:', disabled=False, layout=ip.Layout(width="900px"))
        if len(dsets) > 0: self._dset_selection.value = dsets[0]
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
                                                                          value=self.subsample,
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

    def gui(self) -> ip.HBox:
        return self.getSelectionPanel()

    def getInputFileData( self, vname: str = None, **kwargs ) -> np.ndarray:
        raise NotImplementedError()

    @exception_handled
    def loadDataset(self, **kwargs) -> xa.Dataset:
        lgm().log(f"Load dataset {self.dsid()}, current datasets = {self.datasets.keys()}")
        if self.dsid() not in self.datasets:
            dataset: xa.Dataset = self.loadDataFile(**kwargs)
            vnames = dataset.variables.keys()
            vshapes = [f"{vname}{dataset.variables[vname].shape}" for vname in vnames ]
            lgm().log(f" ---> Opened Dataset {self.dsid()} from file {dataset.attrs['data_file']}\n\t -> variables: {' '.join(vshapes)}")
            if 'plot-x' not in vnames:
                raw_data: xa.DataArray = dataset['norm']      # point data ( shape = [ nsamples, nbands ] )
                dataset['plot-y'] = raw_data
                dataset['plot-x'] = np.arange(0,raw_data.shape[1])
            dataset.attrs['dsid'] = self.dsid()
            dataset.attrs['type'] = 'spectra'
            self.datasets[ self.dsid() ] = dataset
        return self.datasets[ self.dsid() ]

    def blockFilePath( self, **kwargs ) -> str:
        return os.path.join(self.datasetDir, self.dsid(**kwargs) + ".nc")

    def loadDataFile( self, **kwargs ) -> xa.Dataset:
        from spectraclass.data.base import DataManager, dm
        data_file = os.path.join( self.datasetDir, self.dsid(**kwargs) + ".nc" )
        try:
            dataset: xa.Dataset = xa.open_dataset( data_file )
        except FileNotFoundError:
            print( "Preparing input" )
            dm().prepare_inputs()
            dm().save_config()
            dataset: xa.Dataset = xa.open_dataset( data_file )
        dataset.attrs['data_file'] = data_file
        return dataset

    def filterCommonPrefix(self, paths: List[str])-> Tuple[str,List[str]]:
        letter_groups, longest_pre = zip(*paths), ""
        for letter_group in letter_groups:
            if len(set(letter_group)) > 1: break
            longest_pre += letter_group[0]
        plen = len( longest_pre )
        return longest_pre, [ p[plen:] for p in paths ]

    def getDatasetList( self ) -> Tuple[str,List[str]]:
        dset_glob = os.path.expanduser(f"{self.datasetDir}/*.nc")
        lgm().log(f"  Listing datasets from glob: '{dset_glob}' ")
        files = list(filter(os.path.isfile, glob.glob(dset_glob)))
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        filenames = [Path(f).stem for f in files]
        return self.filterCommonPrefix( filenames )

    def loadCurrentProject(self) -> xa.Dataset:
        return self.loadDataset( )

    @property
    def datasetDir(self):
        from spectraclass.data.base import DataManager, dm
        dsdir = os.path.join( self.cache_dir, "spectraclass", self.MODE, dm().name )
        os.makedirs(dsdir, 0o777, exist_ok=True)
        return dsdir




