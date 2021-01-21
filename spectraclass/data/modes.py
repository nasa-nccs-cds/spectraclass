import numpy as np
from typing import List, Optional, Dict
import os, glob, sys
import ipywidgets as ip
from collections import OrderedDict
from spectraclass.reduction.embedding import ReductionManager
from pathlib import Path
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
    subsample = tl.Int(5).tag(config=True, sync=True)
    reduce_method = tl.Unicode("Autoencoder").tag(config=True, sync=True)
    reduce_scope = tl.Unicode("block").tag(config=True, sync=True)
    reduce_nepochs = tl.Int(25).tag(config=True, sync=True)
    reduce_sparsity = tl.Float( 10e-5 ).tag(config=True,sync=True)

    def __init__(self, ):
        super(ModeDataManager,self).__init__()
        assert self.MODE, f"Attempt to instantiate intermediate SingletonConfigurable class: {self.__class__}"
        self.datasets = {}
        self._model_dims_selector: ip.SelectionSlider = None
        self._subsample_selector: ip.SelectionSlider = None
        self._progress = None
        self._dset_selection: ip.Select = None

    @property
    def mode(self):
        if not self.MODE: raise NotImplementedError(f"Mode {self.MODE} has not been implemented")
        return self.MODE.lower()

    def config_scope(self):
        return self.mode.lower()

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

    @property
    def dsid(self) -> str:
        raise NotImplementedError()

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError()

    def updateDatasetList(self):
        if self._dset_selection is not None:
            self._dset_selection.options = self.getDatasetList()

    def select_dataset(self, *args):
        self.dm.select_current_mode()
        if self.dm.dsid != self._dset_selection.value:
            lgm().log( f"Loading dataset '{self._dset_selection.value}', current dataset = '{self.dm.dsid}', "
                   f"current mode = '{self._mode}', current mode index = {self.dm.mode_index}, mdmgr id = {id(self)}")
            self.dm.dsid = self._dset_selection.value
            self.dm.select_dataset(self._dset_selection.value)
        self.dm.refresh_all()

    def getSelectionPanel(self) -> ip.HBox:
        dsets: List[str] = self.getDatasetList()
        self._dset_selection: ip.Select = ip.Select(options=dsets, description='Datasets:', disabled=False)
        if len(dsets) > 0: self._dset_selection.value = dsets[0]
        load: ip.Button = ip.Button(description="Load", border='1px solid dimgrey')
        load.on_click(self.select_dataset)
        filePanel: ip.HBox = ip.HBox([self._dset_selection, load], layout=ip.Layout(width="100%", height="100%"),
                                     border='2px solid firebrick')
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

    def gui(self, **kwargs) -> ip.Tab():
        wTab = ip.Tab(layout=ip.Layout(width='auto', height='auto'))
        selectPanel = self.getSelectionPanel()
        creationPanel = self.getCreationPanel()
        configPanel = self.getConfigPanel()
        wTab.children = [creationPanel, selectPanel, configPanel]
        wTab.set_title(0, "Create")
        wTab.set_title(1, "Select")
        wTab.set_title(2, "Configure")
        return wTab

    def getInputFileData( self, vname: str ) -> np.ndarray:
        raise NotImplementedError()

    def execute_task( self, task: str ):
        from spectraclass.application.controller import app
        if task == "embed":
            app().embed()
        elif task == "mark":
            app().mark()
        elif task == "spread":
            app().spread_selection()
        elif task == "clear":
            app().clear()
        elif task == "undo":
            app().undo_action()
        elif task == "distance":
            app().display_distance()

    @exception_handled
    def loadDataset(self) -> xa.Dataset:
        lgm().log(f"Load dataset {self.dsid}, current datasets = {self.datasets.keys()}")
        if self.dsid not in self.datasets:
            data_file = os.path.join(self.datasetDir, self.dsid + ".nc")
            dataset: xa.Dataset = xa.open_dataset(data_file)
            vnames = dataset.variables.keys()
            vshapes = [f"{vname}{dataset.variables[vname].shape}" for vname in vnames ]
            lgm().log(f" ---> Opened Dataset {self.dsid} from file {data_file}\n\t -> variables: {' '.join(vshapes)}")
            if 'plot-x' not in vnames:
                raw_data: xa.DataArray = dataset['raw']
                dataset['plot-y'] = raw_data
                dataset['plot-x'] = np.arange(0,raw_data.shape[1])
            dataset.attrs['dsid'] = self.dsid
            dataset.attrs['type'] = 'spectra'
            self.datasets[self.dsid] = dataset
        return self.datasets[self.dsid]

    def getDatasetList(self):
        dset_glob = os.path.expanduser(f"{self.datasetDir}/*.nc")
        lgm().log(f"  Listing datasets from glob: '{dset_glob}' ")
        files = list(filter(os.path.isfile, glob.glob(dset_glob)))
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return [Path(f).stem for f in files]

    @property
    def dm(self):
        from .base import DataManager
        return DataManager.instance()

    def loadCurrentProject(self) -> xa.Dataset:
        return self.loadDataset( )

    @property
    def datasetDir(self):
        dsdir = os.path.join( self.cache_dir, self.dm.name, self.MODE )
        os.makedirs(dsdir, exist_ok=True)
        return dsdir




