import numpy as np
from typing import List, Union, Tuple, Optional, Dict
import os, math, pickle, glob
from enum import Enum
import ipywidgets as ip
from functools import partial
from collections import OrderedDict
from spectraclass.reduction.embedding import ReductionManager
from pathlib import Path
import xarray as xa
import traitlets as tl
import traitlets.config as tlc
from spectraclass.model.base import SCConfigurable, AstroModeConfigurable

class ModeDataManager(tlc.Configurable, AstroModeConfigurable):
    MODE = None
    METAVARS = None
    INPUTS = None

    dataset = tl.Unicode("NONE").tag(config=True,sync=True)
    cache_dir = tl.Unicode(os.path.expanduser("~/Development/Cache")).tag(config=True)
    data_dir = tl.Unicode(os.path.expanduser("~/Development/Data")).tag(config=True)

    model_dims = tl.Int(32).tag(config=True, sync=True)
    subsample = tl.Int(5).tag(config=True, sync=True)
    reduce_method = tl.Unicode("Autoencoder").tag(config=True, sync=True)
    reduce_nepochs = tl.Int(1000).tag(config=True, sync=True)

    def __init__(self, ):
        tlc.Configurable.__init__(self)
        AstroModeConfigurable.__init__( self, self.MODE )
        self.datasets = {}
        self._model_dims_selector: ip.SelectionSlider = None
        self._subsample_selector: ip.SelectionSlider = None
        self._progress = None
        self._dset_selection: ip.Select = None

    @property
    def mode(self):
        if not self.MODE: raise NotImplementedError(f"Mode {self.MODE} has not been implemented")
        return self.MODE

    @property
    def metavars(self):
        return self.METAVARS

    def register(self):
        pass

    @classmethod
    def getXarray(cls, id: str, xcoords: Dict, subsample: int, xdims: OrderedDict, **kwargs) -> xa.DataArray:
        from .base import DataManager
        np_data: np.ndarray = DataManager.instance().getInputFileData(id, subsample, tuple(xdims.keys()))
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

    def prepare_inputs(self, *args, **kwargs):
        self.update_gui_parameters()
        self.set_progress(0.02)
        write = kwargs.get('write', True)
        file_name = f"raw" if self.reduce_method == "None" else f"{self.reduce_method}-{self.model_dims}"
        if self.subsample > 1: file_name = f"{file_name}-ss{self.subsample}"
        output_file = os.path.join(self.datasetDir, file_name + ".nc")
        assert (self.INPUTS is not None), f"INPUTS undefined for mode {self.mode}"

        np_embedding: np.ndarray = self.getInputFileData( self.INPUTS['embedding'], self.subsample)
        dims = np_embedding.shape
        mdata_vars = list(self.INPUTS['directory'])
        xcoords = OrderedDict(samples=np.arange(dims[0]), bands=np.arange(dims[1]))
        xdims = OrderedDict({dims[0]: 'samples', dims[1]: 'bands'})
        data_vars = dict( embedding=xa.DataArray(np_embedding, dims=xcoords.keys(), coords=xcoords, name=self.INPUTS['embedding']))
        data_vars.update({vid: self.getXarray(vid, xcoords, self.subsample, xdims) for vid in mdata_vars})
        pspec = self.INPUTS['plot']
        data_vars.update(  {f'plot-{vid}': self.getXarray(pspec[vid], xcoords, self.subsample, xdims, norm=pspec.get('norm','spectral')) for vid in ['x', 'y'] } )
        self.set_progress(0.1)
        if self.reduce_method != "None":
            input_data = data_vars['embedding']
            (reduced_spectra, reproduced_spectra) = ReductionManager.instance().reduce(input_data, self.reduce_method, self.model_dims, self.reduce_nepochs)
            coords = dict(samples=xcoords['samples'], model=np.arange(self.model_dims))
            data_vars['reduction'] = xa.DataArray(reduced_spectra, dims=['samples', 'model'], coords=coords)
            data_vars['reproduction'] = input_data.copy(data=reproduced_spectra)
            self.set_progress(0.8)

        dataset = xa.Dataset(data_vars, coords=xcoords, attrs={'type': 'spectra'})
        dataset.attrs["colnames"] = mdata_vars
        if write:
            print(f"Writing output to {output_file}")
            dataset.to_netcdf(output_file, format='NETCDF4', engine='netcdf4')
        self.updateDatasetList()
        self.set_progress(1.0)
        return dataset

    def updateDatasetList(self):
        if self._dset_selection is not None:
            self._dset_selection.options = self.getDatasetList()

    def select_dataset(self, *args):
        from spectraclass.gui.application import Spectraclass
        self.dm.select_current_mode()
        if self.dm.dataset != self._dset_selection.value:
            print( f"Loading dataset '{self._dset_selection.value}', current dataset = '{self.dm.dataset}', "
                   f"current mode = '{self._mode}', current mode index = {self.dm.mode_index}, mdmgr id = {id(self)}")
            self.dm.dataset = self._dset_selection.value
            self.dm.select_dataset(self._dset_selection.value)
        Spectraclass.instance().refresh_all()

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
            from spectraclass.gui.application import Spectraclass
            rm.nepochs = nepochs_selector.value
            rm.alpha = alpha_selector.value
            rm.init = init_selector.value
            Spectraclass.instance().save_config()

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

    def getInputFileData(self, input_file_id: str, subsample: int = 1, dims: Tuple[int] = None) -> np.ndarray:
        raise NotImplementedError()

    def loadDataset(self, dsid: str, *args, **kwargs) -> xa.Dataset:
        print(f"Load dataset {dsid}, current datasets = {self.datasets.keys()}")
        if dsid is None: return None
        if dsid not in self.datasets:
            data_file = os.path.join(self.datasetDir, dsid + ".nc")
            dataset: xa.Dataset = xa.open_dataset(data_file)
            print(f" ---> Opened Dataset {dsid} from file {data_file}")
            vshapes = [f"{vname}{dataset.variables[vname].shape}" for vname in dataset.variables.keys()]
            print(f"Variables: {', '.join(vshapes)}")
            dataset.attrs['dsid'] = dsid
            dataset.attrs['type'] = 'spectra'
            self.datasets[dsid] = dataset
        return self.datasets[dsid]

    def getDatasetList(self):
        dset_glob = os.path.expanduser(f"{self.datasetDir}/*.nc")
        print(f"  Listing datasets from glob: '{dset_glob}' ")
        files = list(filter(os.path.isfile, glob.glob(dset_glob)))
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return [Path(f).stem for f in files]

    @property
    def dm(self):
        from .base import DataManager
        return DataManager.instance()

    def loadCurrentProject(self) -> xa.Dataset:
        return self.loadDataset(self.dm.dataset)

    @property
    def datasetDir(self):
        dsdir = os.path.join(self.cache_dir, self.dm.name, self.config_mode)
        os.makedirs(dsdir, exist_ok=True)
        return dsdir