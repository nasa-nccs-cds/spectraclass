from typing import List, Optional, Dict, Tuple, Union
import panel as pn
from os import path
import holoviews as hv
from panel.layout import Panel
from pathlib import Path
from holoviews.streams import Stream, param
from spectraclass.gui.control import UserFeedbackManager, ufm
from sklearn.decomposition import PCA, FastICA
import ipywidgets as ip
import xarray as xa
import traitlets as tl
from panel.widgets import Button, Select
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable
import time, numpy as np
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Optional, Dict, Tuple
import os, glob, shutil
from enum  import  Enum

def invert( X: np.ndarray ) -> np.ndarray:
    return X.max() - X

def pnorm(data: xa.DataArray, dim: int = 1) -> xa.DataArray:
    dave, dmag = np.nanmean(data.values, keepdims=True, axis=dim), np.nanstd(data.values, keepdims=True, axis=dim)
    normed_data = (data.values - dave) / dmag
    return data.copy(data=normed_data)

def norm( x: xa.DataArray, axis = 0 ) -> xa.DataArray:
    return ( x - x.data.mean(axis=axis) ) / x.data.std(axis=axis)

def list2str( list: List, sep: str ) -> str:
    slist = [ str(bc) for bc in list ]
    return sep.join( slist )

def scale( x: xa.DataArray, axis = 0 ) -> xa.DataArray:
    result = x / x.mean(axis=axis)
    result.attrs.update( x.attrs )
    return result

def nsamples( trainingsets: List[np.ndarray ]):
    return sum( [ts.shape[0] for ts in trainingsets] )

class BlockSelectMode(Enum):
    LoadTile = 0
    SelectTile = 1
    CreateMask = 2
    LoadMask = 3

ParameterStream = Stream.define('Parameters', value=param.Tuple(default=('Block Mask','NONE'), doc='Display Parameter Value') )

class ModeDataManager(SCSingletonConfigurable):
    from spectraclass.application.controller import SpectraclassController

    MODE = None
    METAVARS = None
    INPUTS = None
    application: SpectraclassController = None

    _image_names = tl.Dict( default_value={} ).tag( config=True, sync=True, cache=False )
    images_glob = tl.Unicode(default_value="").tag( config=True, sync=True, cache=False )
    dset_name = tl.Unicode( "" ).tag(config=True)
    cache_dir = tl.Unicode( "" ).tag(config=True)
    data_dir = tl.Unicode( "" ).tag(config=True)
    class_file = tl.Unicode("NONE").tag(config=True, sync=True)

    model_dims = tl.Int(16).tag(config=True, sync=True)
    subsample_index = tl.Int(1).tag(config=True, sync=True)
    anomaly = tl.Unicode("none").tag(config=True, sync=True)
    modelkey = tl.Unicode("0000").tag(config=True, sync=True)

    def __init__(self, ):
        super(ModeDataManager,self).__init__()
        assert self.MODE, f"Attempt to instantiate intermediate SingletonConfigurable class: {self.__class__}"
        self.datasets = {}
        self._valid_bands = None
        self._current_dataset: Optional[xa.Dataset] = None
        self._model_dims_selector: ip.SelectionSlider = None
        self._samples_axis = None
        self._subsample_selector: ip.SelectionSlider = None
        self._progress: ip.FloatProgress = None
        self._dset_selection: ip.Select = None
        self._dataset_prefix: str = ""
        self._file_selector = None
        self._active_image = 0
        self._autoencoder = None
        self._encoder = None
        self._metadata: Dict = None
        self._spectral_mean: Optional[xa.DataArray] = None
        self.file_selection_watcher = None
        self.parameter_stream: Stream = ParameterStream()
        self.parameter_table = hv.DynamicMap( self.get_parameter_display, streams=[ self.parameter_stream ] )
        self._parameters: Dict = {"Cluster Mask": "NONE", "Block Mask": "NONE"}

    def update_parameter(self, name: str, value ):
        self.parameter_stream.event( value=(name,value) )

    def get_tile_selection_gui(self, **kwargs):
        raise Exception( "get_tile_selection_gui: call to undefined abstact method")

    @exception_handled
    def get_parameter_display(self, value: Tuple) -> hv.Table:
        from spectraclass.gui.control import get_parameter_table
        self._parameters[ value[0] ] = [ value[1] ]
        return get_parameter_table( self._parameters, height=60, width=400 )

    def getSpectralMean(self, norm=False ) -> Optional[xa.DataArray]:
        if self._spectral_mean is None:
            self._spectral_mean = self.load_spectral_mean()
        return tm().norm(self._spectral_mean) if norm else self._spectral_mean

    def load_spectral_mean(self) -> Optional[xa.DataArray]:
        from spectraclass.data.base import DataManager, dm
        file_path = f"{dm().cache_dir}/{self.modelkey}.spectral_mean.nc"
        if os.path.exists( file_path ):
            spectral_mean: xa.DataArray = xa.open_dataarray( file_path )
            return spectral_mean

    @property
    def ext(self):
        return self.images_glob[-4:]

    @property
    def metadata(self) -> Dict:
        if self._metadata is None:
            self._metadata = self.loadMetadata()
        return self._metadata

    def write_metadata(self, block_data, attrs ):
        from spectraclass.data.base import DataManager, dm
        file_path = f"{dm().cache_dir}/{self.modelkey}.mdata.txt"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w") as mdfile:
                for (k, v) in attrs.items():
                    mdfile.write(f"{k}={v}\n")
                for bcoords, bsize in block_data.items():
                    mdfile.write(f"nvalid-{list2str(bcoords,'-')}={bsize}\n")
            lgm().log(f" ---> Writing metadata file at {file_path}", print=True)
        except Exception as err:
            lgm().log(f" ---> ERROR Writing metadata file at {file_path}: {err}", print=True)
            lgm().exception( 'ERROR Writing metadata file' )
            if os.path.isfile(file_path): os.remove(file_path)

    def loadMetadata(self) -> Optional[Dict]:
        from spectraclass.data.base import DataManager, dm
        file_path = f"{dm().cache_dir}/{self.modelkey}.mdata.txt"
        mdata = {}
        try:
            with open(file_path, "r") as mdfile:
                print(f"Loading metadata from file: {file_path}")
                block_sizes = {}
                for line in mdfile.readlines():
                    try:
                        toks = line.split("=")
                        if toks[0].startswith('nvalid'):
                            bcoords = tuple( [ int(iv) for iv in toks[0].split("-")[1:] ] )
                            block_sizes[bcoords] = int(toks[1])
                        else:
                            mdata[toks[0]] = toks[1]
                    except Exception as err:
                        lgm().log(f"LoadMetadata: Error '{err}' reading line '{line}'")
                mdata['block_size'] = block_sizes
                return mdata
        except Exception as err:
            lgm().log(f"Warning: can't read config file '{file_path}': {err}\n")
        return None


    @property
    def block_sizes(self) -> Dict[Tuple[int,int,int], int]:
        return self.metadata['block_size']

    def block_nvalid(self, block_coords: Tuple[int,int,int]) -> int:
        return self.block_sizes.get(tuple(block_coords), 0)

    def get_valid_block_coords(self, image_index, block_coords: Tuple[int,int,int]) -> Tuple[int,int,int]:
        from spectraclass.data.base import DataManager, dm
        if self.block_nvalid(block_coords) > 0: return block_coords
        for (coords, nvalid) in self.block_sizes.items():
            if (image_index==coords[0]) and (nvalid > 0): return coords
        lgm().log(f"No valid blocks in tile.\nMetadata File: {dm().metadata_file}\nBlock sizes: {self.block_sizes}")
        raise Exception("No valid blocks in tile")

    # def spectral_reduction(data, graph, n_components=3, sparsify=False):
    #     t0 = time.time()
    #     graph = graph.tocoo()
    #     graph.sum_duplicates()
    #     if sparsify:
    #         n_epochs = 200
    #         graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    #         graph.eliminate_zeros()
    #
    #     random_state = np.random.RandomState()
    #     initialisation = spectral_layout(data, graph, n_components, random_state, metric="euclidean")
    #     expansion = 10.0 / np.abs(initialisation).max()
    #     rv = (initialisation * expansion).astype(np.float32)
    #     print(f"Completed spectral_embedding in {(time.time() - t0) / 60.0} min.")
    #     return rv

    def ca_reduction(self, train_input: xa.DataArray, ndim: int, method = "pca" ) -> Tuple[np.ndarray, np.ndarray]:
        if method == "pca":
            mapper = PCA(n_components=ndim)
        elif method == "ica":
            mapper = FastICA(n_components=ndim)
        else: raise Exception( f"Unknown reduction methos: {method}")
        normed_train_input: xa.DataArray = norm( train_input  )
        mapper.fit( normed_train_input.data )
        if method == "pca":
            lgm().log( f"PCA reduction[{ndim}], Percent variance explained: {mapper.explained_variance_ratio_ * 100}" )
            reduced_features: np.ndarray = mapper.transform( normed_train_input.data )
            reproduction: np.ndarray = mapper.inverse_transform(reduced_features)
            return ( reduced_features, reproduction )
        else:
            normed_input = norm( train_input )
            reduced_features: np.ndarray = mapper.transform( normed_input.data )
            reproduction: np.ndarray = mapper.inverse_transform(reduced_features)
            return (reduced_features, reproduction )

    # def _load_network(self, **kwargs  ) -> bool:
    #     aefiles = self.autoencoder_files( **kwargs, withext=True  )
    #     if not os.path.exists( aefiles[0] ):
    #         lgm().log( f"#AEC model file does not exist: {aefiles[0]}")
    #         return False
    #     else:
    #         lgm().log( f"#AEC: LOADING ENCODER from '{aefiles}'")
    #         self._autoencoder = keras_load_model(  f"{aefiles[0]}" )
    #         self._encoder =     keras_load_model(  f"{aefiles[1]}" )
    #         return True

    # def get_trained_network(self, **kwargs ):
    #     if self._autoencoder is None:
    #         self.autoencoder_preprocess( refresh_model=False, **kwargs )

    def load_weights(self, **kwargs) -> bool:
        aefiles = self.autoencoder_files( **kwargs )
        wfile = aefiles[0] + ".index"
        if self.refresh_model:
            lgm().log(f"#AEC refreshing weights")
            return False
        elif not os.path.exists( wfile ):
            lgm().log( f"#AEC weights file does not exist: {wfile}")
            return False
        else:
            try:
                self._autoencoder.load_weights( aefiles[0] )
                self._encoder.load_weights( aefiles[1] )
                lgm().log(f"#AEC: Loaded saved weights.")
                return True
            except Exception as err:
                lgm().log(f"#AEC: Unable to load saved weights for files {aefiles}: {err}")
                return False

    def autoencoder_files(self, **kwargs ) -> List[str]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.base import DataManager, dm
        key: str = kwargs.get( 'key', self.modelkey )
        filter_sig = tm().get_band_filter_signature()
        model_dims: int = kwargs.get('dims', self.model_dims)
        aefiles = [f"{dm().cache_dir}/autoencoder.{model_dims}.{filter_sig}.{key}.{dm().modal.anomaly}", f"{dm().cache_dir}/encoder.{model_dims}.{filter_sig}.{key}"]
        lgm().log(f"#AEC: autoencoder_files (key={key}): {aefiles}")
        return aefiles

    def initialize_dimension_reduction( self, **kwargs ):
        lgm().log( "AEC: initialize_dimension_reduction" )
        self.prepare_inputs( **kwargs )

    # def autoencoder_preprocess(self, **kwargs ):
    #     niter: int = kwargs.get( 'niter', self.reduce_niter )
    #     method: str = kwargs.get( 'method', self.reduce_method )
    #     dropout: float = kwargs.get('dropout', self.reduce_dropout)
    #     input_data: Optional[xa.DataArray] = tm().getBlock().data
    #     if input_data is not None:
    #         input_dims = input_data.shape[0]
    #         lr = kwargs.get('lr', self.reduce_learning_rate )
    #         self.vae = (method.strip().lower() == 'vae')
    #         aefiles = self.autoencoder_files(**kwargs)
    #         lgm().log(f"#AEC autoencoder_preprocess, aefiles = {aefiles}" )
    #         if self.refresh_model:
    #             for aef in aefiles:
    #                 for ifile in glob.glob(aef + ".*"): os.remove(ifile)
    #         self.build_encoder( input_dims, dropout=dropout, lr=lr, **kwargs )
    #         weights_loaded = self.load_weights(**kwargs)
    #         initial_epoch = 0
    #         if not weights_loaded:
    #             for iter in range(niter):
    #                 initial_epoch = self.general_training( initial_epoch, **kwargs )
    #                 if self.reduce_focus_nepoch > 0:
    #                     initial_epoch = self.focused_training( initial_epoch, **kwargs )
    #             self._autoencoder.save_weights( aefiles[0] )
    #             self._encoder.save_weights( aefiles[1] )
    #             lgm().log(f"#AEC autoencoder_preprocess completed, saved model weights to files={aefiles}", print=True)
    #
    # def general_training(self, initial_epoch = 0, **kwargs ):
    #     from spectraclass.data.base import DataManager, dm
    #     from spectraclass.data.spatial.tile.tile import Block, Tile
    #     num_reduce_images = min( dm().modal.num_images, self.reduce_nimages )
    #     for image_index in range( num_reduce_images ):
    #         dm().modal.set_current_image(image_index)
    #         blocks: List[Block] = tm().tile.getBlocks()
    #         num_training_blocks = min( self.reduce_nblocks, len(blocks) )
    #         lgm().log(f"Autoencoder general training: {num_training_blocks} blocks for image[{image_index}/{num_reduce_images}]: {dm().modal.image_name}", print=True)
    #         lgm().log(f" NBLOCKS = {self.reduce_nblocks}/{len(blocks)}, block shape = {blocks[0].shape}")
    #         for iB, block in enumerate(blocks):
    #             if iB < self.reduce_nblocks:
    #                 t0 = time.time()
    #                 norm_point_data, grid = block.getPointData( norm=True )
    #                 if norm_point_data.shape[0] > 0:
    #                     final_epoch = initial_epoch + self.reduce_nepoch
    #                     lgm().log( f" ** ITER[{iter}]: Processing block{block.block_coords}, norm data shape = {norm_point_data.shape}", print=True)
    #                     history: tf.keras.callbacks.History = self._autoencoder.fit(norm_point_data.data, norm_point_data.data, initial_epoch=initial_epoch,
    #                                                              epochs=final_epoch, batch_size=256, shuffle=True)
    #                     initial_epoch = final_epoch
    #                     lgm().log(f" Trained autoencoder in {time.time() - t0} sec", print=True)
    #                 block.initialize()
    #     return initial_epoch
    #


    def get_anomaly(self, train_data: np.ndarray, reproduced_data ) -> np.ndarray:
        return np.abs(train_data - reproduced_data).sum(axis=-1, keepdims=False)

    def get_anomaly_threshold(self, anomaly: np.ndarray, anom_focus=0.10 ) -> float:
        hist, edges = np.histogram(anomaly, 64)
        counts: np.ndarray = np.cumsum(hist)
        ti: int = np.abs(counts - counts[-1] * (1 - anom_focus)).argmin()
        return edges[ti + 1]

    def autoencoder_reduction(self, train_input: xa.DataArray, **kwargs ) -> Tuple[xa.DataArray,xa.DataArray]:
        from spectraclass.reduction.trainer import mt
        ufm().show("Computing Feature Space...")
        t0 = time.time()
        ispecs: List[np.ndarray] = [train_input.data.max(0), train_input.data.min(0), train_input.data.mean(0), train_input.data.std(0)]
        lgm().log(f" autoencoder_reduction: train_input shape = {train_input.shape} ")
        lgm().log(f"   ----> max = { ispecs[0][:64].tolist() } ")
        lgm().log(f"   ----> min = { ispecs[1][:64].tolist() } ")
        lgm().log(f"   ----> ave = { ispecs[2][:64].tolist() } ")
        lgm().log(f"   ----> std = { ispecs[3][:64].tolist() } ")
        encoded_data: xa.DataArray  = mt().encode( train_input )
        reproduced_data: xa.DataArray = mt().predict( train_input )
        lgm().log(f" Autoencoder_reduction, result shape = {encoded_data.shape}")
        lgm().log(f" ----> encoder_input: shape = {train_input.shape}")
        lgm().log(f" ----> reproduction: shape = {reproduced_data.shape}")
        lgm().log(f" ----> encoding: shape = {encoded_data.shape}, std = {encoded_data.std()} ")
#        anomaly = np.abs( train_input.values - reproduced_data ).sum( axis=-1, keepdims=False )
#        dmask = anomaly > 0.0
#        lgm().log( f" ----> ANOMALY: shape = {anomaly.shape}, range = [{anomaly.min(where=dmask,initial=np.inf)},{anomaly.max()}] ")
        ufm().show(f"Done Computing Features in {time.time()-t0:.2f} sec")
        return (encoded_data, reproduced_data)

    @property
    def image_names(self) -> Dict[str,str]:
        self.generate_image_list()
        return self._image_names

    @property
    def extension(self):
        return self.ext

    @property
    def default_images_glob(self):
        return "*" + self.extension

    @exception_handled
    def generate_image_list(self):
        if len( self._image_names ) == 0:
            image_path_list = self.get_image_paths()
            lgm().log(f" ---> FOUND {len(image_path_list)} paths")
            self._image_names = { self.extract_image_name( image_path ): image_path for image_path in image_path_list }
            lgm().log( f" ---> IMAGE LIST: {self._image_names}")

    @exception_handled
    def get_image_paths(self) -> List[str]:
        lgm().log(f"generate_image_list")
        iglob = f"{self.data_dir}/{(self.images_glob if self.images_glob else self.default_images_glob)}"
        lgm().log(f" ---> glob: '{iglob}'")
        return glob.glob(iglob)

    def set_current_image(self, image_index: int ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if image_index != self._active_image:
            self._active_image = image_index
            self._current_dataset = None
            msg = f"Setting active image[{self._active_image}]: {self.image_name}"
            ufm().show( msg )
            tm().tile.initialize()

    @classmethod
    def extract_image_name( cls, image_path: str ) -> str:
        name = Path(image_path).stem
        return name

    @property
    def num_images(self):
        return len( self.image_names )

    @property
    def image_index(self) -> int:
        return self._active_image

    @property
    def image_name(self):
        inames = list(self.image_names.keys())
        return inames[self._active_image]

    def image_path(self, local_image_path: str ) -> str:
        base_image_name = local_image_path.split('/')[0]
        image_name = base_image_name.replace("ang","").replace("rfl","")
        return self.image_names[image_name]

    def get_image_name( self, image_index: int ):
        inames = list(self.image_names.keys())
        return inames[ image_index ]

    @property
    def file_selector(self):
        if self._file_selector is None:
            inames = list( self.image_names.keys() )
            self._file_selector =  pn.widgets.Select(name='Image', options=inames, value=inames[0] )
            self.file_selection_watcher = self.file_selector.param.watch(self.on_image_change, ['value'], onlychanged=True)
        return self._file_selector

    def on_image_change(self,*events):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.spatial.map import MapManager, mm
        for event in events:
            if event.name == 'value':
                self.set_current_image(self.file_selector.index)
                dm().clear_project_cache()
                dm().modal.update_extent()
                mm().update()

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

    def get_block_selection(self,**kwargs) -> Optional[Dict]:
        return None

    def register(self):
        pass

    def save_block_selection(self):
        pass

    def valid_bands(self) -> Optional[List]:
        return self._valid_bands

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

    def process_block(self, block, has_metadata: bool  ) -> xa.Dataset:
        raise NotImplementedError()

    def reduce(self, train_data: xa.DataArray, **kwargs ) -> Tuple[xa.DataArray,xa.DataArray]:
        with xa.set_options(keep_attrs=True):
            return  self.autoencoder_reduction( train_data, **kwargs )

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

    def getSelectionPanel(self,**kwargs) -> pn.Column:
        from spectraclass.data.base import DataManager, dm
        self._dataset_prefix, dsets = self.getDatasetList()
        self._dset_selection: Select = Select( options=dsets, description='Datasets:', disabled=False )
        if len(dsets) > 0: self._dset_selection.value = dm().dsid()[ len(self._dataset_prefix): ]
        load: Button = Button(description="Load", border='1px solid dimgrey')
        load.on_click( self.selected_dataset )
        filePanel: pn.Column = pn.Column( self._dset_selection, load )
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

    def getCreationPanel(self) -> Panel:
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
        return self.getSelectionPanel(**kwargs)

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
        return dvars

    @exception_handled
    def loadDataset(self, **kwargs) -> Optional[ Dict[str,Union[xa.DataArray,List,Dict]] ]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.spatial.tile.tile import Block
        from spectraclass.data.base import DataManager, dm, DataType
        lgm().log(f"Load dataset, current = {list(self.datasets.keys())}")
        self._current_dataset = None
        if self.dsid() not in self.datasets:
            lgm().log(f"Load dataset {self.dsid()}, current datasets = {list(self.datasets.keys())}")
            if not dm().refresh_data:
                self._current_dataset = self.loadDataFile(**kwargs)
            block: Block = tm().getBlock(**kwargs)
            if self._current_dataset is None:
                has_metadata = (self.metadata is not None)
                self._current_dataset = self.process_block( block, has_metadata )
            if (self._current_dataset is None) or (len(self._current_dataset.variables.keys()) == 0):
                lgm().log(f"Warning: Attempt to Load empty dataset {self.dataFile( **kwargs )}")
                return None
            else:
                lgm().log(f" ---> Opening Dataset {self.dsid()}")
                dvars: Dict[str,Union[xa.DataArray,List,Dict]] = self.dset_subsample( self._current_dataset, dsid=self.dsid(), **kwargs )
                attrs = self._current_dataset.attrs.copy()
                raw_data = dvars['raw']
                point_data, pcoords = block.getPointData()
                lgm().log( f" -----> point_data: shape = {raw_data.shape}, #NULL={np.count_nonzero(np.isnan(raw_data.values))}/{raw_data.size}")
                dvars['samples'] = point_data.coords['samples']
                attrs['dsid'] = self.dsid()
                attrs['type'] = 'spectra'
                dvars['attrs'] = attrs
                self.datasets[ self.dsid() ] = dvars
        return self.datasets[ self.dsid() ]

    def blockFilePath( self, **kwargs ) -> str:
        ext = kwargs.get('../ext', 'nc')
        return path.join(self.datasetDir, self.dsid(**kwargs) + "." + ext )

    def removeDataset(self):
        lgm().log( "Removing existing block data files.", print=True )
        for f in os.listdir(self.datasetDir):
            file = os.path.join(self.datasetDir, f)
            if path.isfile( file ):
                try: os.remove( file )
                except: pass

    def leafletRasterPath( self, **kwargs ) -> str:
        from spectraclass.data.base import DataManager, dm
        return f"files/spectraclass/datasets/{self.MODE}/{dm().name}/{self.dsid(**kwargs)}{self.ext}"

    def dataFile( self, **kwargs ):
        raise NotImplementedError( "Attempt to call virtual method")

    def hasBlockData(self) -> bool:
        return path.isfile( self.dataFile() )

    def loadDataFile( self, **kwargs ) -> Optional[xa.Dataset]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block = kwargs.get( 'block', tm().getBlock() )
        ufm().show(f"Loading Block: {block.block_coords} ")
        if (self._current_dataset is None) or (self._current_dataset.attrs['block_coords'] != block.block_coords ):
            dFile = self.dataFile( **kwargs )
            if path.isfile( dFile ):
                lgm().log(f"#GID: Loading Block {block.block_coords} ")
                self._current_dataset = xa.open_dataset( dFile, concat_characters=True )
                self._current_dataset.attrs['data_file'] = dFile
                self._current_dataset.attrs['block_coords'] = block.block_coords
                vars = [ f"{vid}{var.dims}" for (vid,var) in self._current_dataset.variables.items()]
                coords = [f"{cid}{coord.shape}" for (cid, coord) in self._current_dataset.coords.items()]
                lgm().log( f"#GID: loadDataFile: {dFile}, coords={coords}, vars={vars}" )
                lgm().log( f"#GID:  --> coords={coords}")
                lgm().log( f"#GID:  --> vars={vars}")
                x,y = self._current_dataset.x.values, self._current_dataset.y.values
                lgm().log(f"#GID:  --> exent= ({x[0]},{x[-1]}) ({y[0]},{y[-1]})")
        return self._current_dataset

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

    def loadCurrentProject( self, **kwargs ) -> Optional[ Dict[str,Union[xa.DataArray,List,Dict]] ]:
        return self.loadDataset( **kwargs )

    @property
    def datasetDir(self):
        from spectraclass.data.base import DataManager, dm
        dsdir = path.join( self.cache_dir, "spectraclass", dm().mode )
        os.makedirs(dsdir, 0o777, exist_ok=True)
        return dsdir

    def readSpectralData(self, **kwargs) -> xa.DataArray:
        raise NotImplementedError( "Attempt to call abstract method 'readSpectralData'")




