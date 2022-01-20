import numpy as np
from typing import List, Optional, Dict, Tuple
import ipywidgets as ipw
import os, glob, sys
import ipywidgets as ip
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

    image_names = tl.List( tl.Unicode(), ["NONE"] ).tag(config=True ,sync=True)
    cache_dir = tl.Unicode(os.path.expanduser("~/Development/Cache")).tag(config=True)
    data_dir = tl.Unicode(os.path.expanduser("~/Development/Data")).tag(config=True)
    class_file = tl.Unicode("NONE").tag(config=True, sync=True)

    model_dims = tl.Int(32).tag(config=True, sync=True)
    subsample = tl.Int(1).tag(config=True, sync=True)
    reduce_method = tl.Unicode("Autoencoder").tag(config=True, sync=True)
    reduce_nepochs = tl.Int(5).tag(config=True, sync=True)
    reduce_sparsity = tl.Float( 0.0 ).tag(config=True,sync=True)

    def __init__(self, ):
        super(ModeDataManager,self).__init__()
        assert self.MODE, f"Attempt to instantiate intermediate SingletonConfigurable class: {self.__class__}"
        self.datasets = {}
        self._model_dims_selector: ip.SelectionSlider = None
        self._subsample_selector: ip.SelectionSlider = None
        self._progress: ip.FloatProgress = None
        self._dset_selection: ip.Select = None
        self._dataset_prefix: str = ""
        self._file_selector = None
        self._image_name = None

    @property
    def image_name(self):
        if self._image_name is None:
            self._image_name = self.image_names[0]
        return self._image_name

    @property
    def file_selector(self):
        if self._file_selector is None:
            lgm().log( f"Creating file_selector, options={self.image_names}, value={self.image_names[0]}")
            self._file_selector =  ip.Select( options=self.image_names, value=self.image_name, layout=ipw.Layout(width='600px') )
            self._file_selector.observe( self.on_image_change, names=['value'] )
        return self._file_selector

    def on_image_change( self, event: Dict ):
        from spectraclass.gui.spatial.map import MapManager, mm
        self._image_name = self.file_selector.value
        mm().update_plots( new_image=self._image_name )

    def set_image_name(self, image_name: str ):
        from spectraclass.gui.spatial.map import MapManager, mm
        self._image_name = image_name
        lgm().log( f"Install new image: {image_name}", print=True )
        mm().update_plots( new_image=True )

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
        if dm().dsid() != self.selected_dataset:
            ufm().show( "Loading new data block")
            lgm().log( f"Loading dataset '{self.selected_dataset}', current dataset = '{dm().dsid()}', mdmgr id = {id(self)}")
            dm().loadProject( self.selected_dataset )
            ufm().clear()
        dm().refresh_all()

    def getSelectionPanel(self) -> ip.HBox:
        from spectraclass.data.base import DataManager, dm
        # self._dataset_prefix, dsets = self.getDatasetList()
        # self._dset_selection: ip.Select = ip.Select(options=dsets, description='Datasets:', disabled=False, layout=ip.Layout(width="900px"))
        # if len(dsets) > 0: self._dset_selection.value = dm().dsid()[ len(self._dataset_prefix): ]
        # load: ip.Button = ip.Button(description="Load", border='1px solid dimgrey')
        # load.on_click(self.select_dataset)
        # filePanel: ip.HBox = ip.HBox([self._dset_selection, load], layout=ip.Layout(width="100%", height="100%"), border='2px solid firebrick')
        # return filePanel
        return ip.HBox([])

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

    def gui( self, **kwargs ):
        pass
#       return self.getSelectionPanel()

    def getInputFileData( self, vname: str = None, **kwargs ) -> np.ndarray:
        raise NotImplementedError()

    @exception_handled
    def loadDataset(self, **kwargs) -> Optional[xa.Dataset]:
        if self.dsid() not in self.datasets:
            lgm().log(f"Load dataset {self.dsid()}, current datasets = {self.datasets.keys()}")
            dataset: xa.Dataset = self.loadDataFile(**kwargs)
            if len(dataset.variables.keys()) == 0:
                lgm().log(f"Warning: Attempt to Load empty dataset {self.dataFile( **kwargs )}", print=True)
                return None
            else:
                vnames = dataset.variables.keys()
                vshapes = [ f"{vname}{dataset.variables[vname].shape}" for vname in vnames ]
                lgm().log(f" ---> Opened Dataset {self.dsid()} from file {dataset.attrs['data_file']}\n\t -> variables: {' '.join(vshapes)}")
                if 'plot-x' not in vnames:
                    raw_data: xa.DataArray = dataset['norm']      # point data ( shape = [ nsamples, nbands ] )
                    model_data: xa.DataArray = dataset['reduction']
                    dataset['plot-y'] = raw_data
                    dataset['plot-x'] = np.arange(0,raw_data.shape[1])
                    dataset['plot-mx'] = np.arange(0, model_data.shape[1])
                dataset.attrs['dsid'] = self.dsid()
                dataset.attrs['type'] = 'spectra'
                self.datasets[ self.dsid() ] = dataset
        return self.datasets[ self.dsid() ]

    def blockFilePath( self, **kwargs ) -> str:
        ext = kwargs.get('ext','nc')
        return os.path.join(self.datasetDir, self.dsid(**kwargs) + "." + ext )

    def leafletRasterPath( self, **kwargs ) -> str:
        from spectraclass.data.base import DataManager, dm
        return f"files/spectraclass/datasets/{self.MODE}/{dm().name}/{self.dsid(**kwargs)}.tif"

    def dataFile( self, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block = kwargs.get('block', tm().getBlock( **kwargs ) )
        return os.path.join( self.datasetDir, block.file_name + f"-m{self.model_dims}.nc" )

    def hasBlockData(self) -> bool:
        return os.path.isfile( self.dataFile() )

    def loadDataFile( self, **kwargs ) -> Optional[xa.Dataset]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        dFile = self.dataFile( **kwargs )
        if os.path.isfile( dFile ):
            lgm().log( f"loadDataFile: {dFile}" )
            dataset: xa.Dataset = xa.open_dataset( dFile )
            dataset.attrs['data_file'] = dFile
        else:
            raise Exception( f"BLOCK[{tm().getBlock().block_coords}]: Missing data file: {dFile}" )
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
        dsdir = os.path.join( self.cache_dir, "spectraclass", dm().mode )
        os.makedirs(dsdir, 0o777, exist_ok=True)
        return dsdir




