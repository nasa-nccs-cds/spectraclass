from typing import List, Optional, Dict, Tuple, Union
import ipywidgets as ipw
import ipywidgets as ip
from os import path
from pathlib import Path
from spectraclass.gui.control import UserFeedbackManager, ufm
from sklearn.decomposition import PCA, FastICA
import tensorflow as tf
import xarray as xa
import traitlets as tl
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable
import time, numpy as np
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Optional, Dict, Tuple
import os, glob, shutil

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

def get_optimizer( **kwargs ):
    oid = kwargs.get( 'optimizer','rmsprop').lower()
    lr = kwargs.get( 'lr', 1e-3 )
    if   oid == "rmsprop": return tf.keras.optimizers.RMSprop( learning_rate=lr )
    elif oid == "adam":    return tf.keras.optimizers.Adam(    learning_rate=lr )
    elif oid == "sgd":     return tf.keras.optimizers.SGD(     learning_rate=lr )
    else: raise Exception( f" Unknown optimizer: {oid}")

def vae_loss( inputs, outputs, n_features, z_mean, z_log ):
    """ Loss = Recreation loss + Kullback-Leibler loss
    for probability function divergence (ELBO).
    gamma > 1 and capacity != 0 for beta-VAE
    """
    gamma = 1.0
    capacity = 0.0
    reconstruction_loss = mse( inputs, outputs )
    reconstruction_loss *= n_features
    kl_loss = 1 + z_log - tf.keras.backend.K.square(z_mean) - tf.keras.backend.K.exp(z_log)
    kl_loss = -0.5 * tf.keras.backend.K.sum(kl_loss, axis=-1)
    kl_loss = gamma * tf.keras.backend.K.abs( kl_loss - capacity )
    return tf.keras.backend.K.mean(reconstruction_loss + kl_loss)

class ModeDataManager(SCSingletonConfigurable):
    from spectraclass.application.controller import SpectraclassController

    MODE = None
    METAVARS = None
    INPUTS = None
    VALID_BANDS = None
    application: SpectraclassController = None

    _image_names = tl.List( default_value=[] ).tag( config=True, sync=True, cache=False )
    images_glob = tl.Unicode(default_value="").tag( config=True, sync=True, cache=False )
    dset_name = tl.Unicode( "" ).tag(config=True)
    cache_dir = tl.Unicode( "" ).tag(config=True)
    data_dir = tl.Unicode( "" ).tag(config=True)
    class_file = tl.Unicode("NONE").tag(config=True, sync=True)

    model_dims = tl.Int(16).tag(config=True, sync=True)
    subsample_index = tl.Int(1).tag(config=True, sync=True)
    reduce_method = tl.Unicode("vae").tag(config=True, sync=True)
    reduce_anom_focus = tl.Float( 0.25 ).tag(config=True, sync=True)
    reduce_nepoch = tl.Int(5).tag(config=True, sync=True)
    reduce_nimages = tl.Int(100).tag(config=True, sync=True)
    reduce_nblocks = tl.Int(250).tag(config=True, sync=True)
    reduce_dropout = tl.Float( 0.01 ).tag(config=True, sync=True)
    reduce_learning_rate = tl.Float(1e-3).tag(config=True, sync=True)
    reduce_focus_nepoch = tl.Int(20).tag(config=True, sync=True)
    reduce_focus_ratio = tl.Float(2.0).tag(config=True, sync=True)
    reduce_niter = tl.Int(1).tag(config=True, sync=True)
    reduce_sparsity = tl.Float( 0.0 ).tag(config=True,sync=True)
    modelkey = tl.Unicode("0000").tag(config=True, sync=True)
    refresh_model = tl.Bool(False).tag(config=True, sync=True)

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
        self._autoencoder = None
        self._encoder = None
        self._metadata: Dict = None

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

    def get_trained_network(self, **kwargs ):
        if self._autoencoder is None:
            self.autoencoder_preprocess( refresh_model=False, **kwargs )

    def build_encoder(self, **kwargs):
        input_dims = kwargs.pop( 'bands', None )
        if input_dims is None: input_dims = tm().getBlock().data.shape[0]
        lgm().log( f"build_encoder, input_dims={input_dims}, parms={kwargs}")
        if self.vae:
            self._build_vae_model( input_dims, **kwargs)
        else:
            self._build_ae_model( input_dims, **kwargs)

    def load_weights(self, **kwargs) -> bool:
        aefiles = self.autoencoder_files(**kwargs)
        wfile = aefiles[0] + ".index"
        if self.refresh_model:
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

    def _build_ae_model(self, input_dims: int, **kwargs):
        model_dims: int = kwargs.pop('dims', self.model_dims)
        verbose = kwargs.pop('verbose', True)
        dropout_rate = kwargs.get('dropout', 0.01)
        loss = str( kwargs.pop('loss', 'mean_squared_error') ).lower()  # mean_squared_error categorical_crossentropy
        lgm().log(f"#AEC: RM BUILD AEC NETWORK: {input_dims} -> {model_dims}")
        winit: float = kwargs.get( 'winit', 0.001 )
        reduction_factor = 2
        inputlayer = tf.keras.layers.Input( shape=[input_dims] )
        activation = kwargs.get( 'activation', 'tanh' )
        optimizer =  get_optimizer( **kwargs )
        dargs = dict( kernel_initializer=tf.keras.initializers.RandomNormal(stddev=winit), bias_initializer=tf.keras.initializers.Zeros() )
        layer_dims, layer = int(round(input_dims / reduction_factor)), inputlayer
        while layer_dims > model_dims:
            layer = tf.keras.layers.Dense(layer_dims, activation=activation, **dargs )(layer)
            if dropout_rate > 0.0: layer = tf.keras.layers.Dropout(dropout_rate)(layer)
            layer_dims = int(round(layer_dims / reduction_factor))
        encoded = x = tf.keras.layers.Dense(model_dims, activation=activation, **dargs )(layer)
        layer_dims = int(round(model_dims * reduction_factor))
        while layer_dims < input_dims:
            layer = tf.keras.layers.Dense(layer_dims, activation=activation, **dargs )(layer)
            if dropout_rate > 0.0: layer = tf.keras.layers.Dropout(dropout_rate)(layer)
            layer_dims = int(round(layer_dims * reduction_factor))
        decoded = tf.keras.layers.Dense(input_dims, activation='linear', **dargs )(layer)
        #        modelcheckpoint = ModelCheckpoint('xray_auto.weights', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        #        earlystopping = EarlyStopping(monitor='loss', min_delta=0., patience=100, verbose=1, mode='auto')
        self._autoencoder = tf.keras.models.Model(inputs=[inputlayer], outputs=[decoded])
        self._encoder = tf.keras.models.Model(inputs=[inputlayer], outputs=[encoded])
        if verbose:
            self._autoencoder.summary()
            self._encoder.summary()
        self._autoencoder.compile( loss=loss, optimizer=optimizer )
        lgm().log(f" BUILD Autoencoder network: input_dims = {input_dims} ")

    def sampling(self, args):

        """Reparametrisation by sampling from Gaussian, N(0,I)
        To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
        with latent variables z: z = z_mean + sqrt(var) * epsilon

        Parameters
        ----------
        args : tensor
            Mean and log of variance of Q(z|X).

        Returns
        -------
        z : tensor
            Sampled latent variable.
        """
        z_mean, z_log = args
        batch = tf.keras.backend.K.shape(z_mean)[0]  # batch size
        dim = tf.keras.backend.K.int_shape(z_mean)[1]  # latent dimension
        epsilon = tf.keras.backend.K.random_normal(shape=(batch, dim))  # mean=0, std=1.0
        return z_mean + tf.keras.backend.K.exp(0.5 * z_log) * epsilon

    def _build_vae_model(self, input_dims: int, **kwargs ):
        model_dims: int = kwargs.get('dims', self.model_dims)
        hidden_activation = kwargs.get('activation',"tanh")
        output_activation = 'linear'
        optimizer = kwargs.get('optimizer','rmsprop')  # 'rmsprop'
        reduction_factor = 2.0
        verbose = 0
        dropout_rate = kwargs.get('dropout',0.01)
        l2_regularizer = kwargs.get('regularizer',0.01)
        inputs = tf.keras.layers.Input(shape=(input_dims,))
        layer = tf.keras.layers.Dense(input_dims, activation=hidden_activation)(inputs)
        layer_dims = int(round(input_dims / reduction_factor))
        while layer_dims > model_dims:
            layer = tf.keras.layers.Dense(layer_dims, activation=hidden_activation, activity_regularizer=tf.keras.regularizers.l2(l2_regularizer))(layer)
            layer = tf.keras.layers.Dropout(dropout_rate)(layer)
            layer_dims = int(round(layer_dims / reduction_factor))

        z_mean = tf.keras.layers.Dense(model_dims)(layer)
        z_log = tf.keras.layers.Dense(model_dims)(layer)
        z = tf.keras.layers.Lambda(self.sampling, output_shape=(model_dims,))([z_mean, z_log])

        self._encoder = tf.keras.models.Model(inputs, [z_mean, z_log, z])
        if verbose >= 1: self._encoder.summary()

        latent_inputs = tf.keras.layers.Input(shape=(model_dims,))
        layer = tf.keras.layers.Dense(model_dims, activation=hidden_activation)(latent_inputs)
        layer_dims = int(round(model_dims * reduction_factor))
        while layer_dims < input_dims:
            layer = tf.keras.layers.Dense(layer_dims, activation=hidden_activation)(layer)
            layer = tf.keras.layers.Dropout(dropout_rate)(layer)
            layer_dims = int(round(layer_dims * reduction_factor))

        outputs = tf.keras.layers.Dense(input_dims, activation=output_activation)(layer)

        self._decoder = tf.keras.models.Model(latent_inputs, outputs)
        if verbose >= 1: self._decoder.summary()
        outputs = self._decoder(self._encoder(inputs)[2])
        self._autoencoder = tf.keras.models.Model(inputs, outputs)
        self._autoencoder.add_loss( vae_loss( inputs, outputs, input_dims, z_mean, z_log ) )
        self._autoencoder.compile( optimizer=optimizer )
        self._decoder.compile( optimizer=optimizer )
        if verbose >= 1: self._autoencoder.summary()
        lgm().log(f"#AEC: RM BUILD VAE NETWORK: {input_dims} -> {model_dims}")

    def autoencoder_files(self, **kwargs ) -> List[str]:
        key: str = kwargs.get( 'key', self.modelkey )
        model_dims: int = kwargs.get('dims', self.model_dims)
        from spectraclass.data.base import DataManager, dm
        aefiles = [f"{dm().cache_dir}/autoencoder.{model_dims}.{key}", f"{dm().cache_dir}/encoder.{model_dims}.{key}"]
        lgm().log(f"#AEC: autoencoder_files (key={key}): {aefiles}")
        return aefiles

    def initialize_dimension_reduction( self, **kwargs ):
        lgm().log( "AEC: initialize_dimension_reduction" )
        self.prepare_inputs( **kwargs )

    def autoencoder_process(self, point_data: xa.DataArray, **kwargs ):
        nepoch: int = kwargs.get( 'nepoch', self.reduce_nepoch )
        dropout: float = kwargs.get('dropout', self.reduce_dropout)
        if self._autoencoder is None:
            method: str = kwargs.pop('method', self.reduce_method)
            self.vae = (method.strip().lower() == 'vae')
            self.build_encoder( dropout=dropout, **kwargs )
        weights_loaded = self.load_weights(**kwargs)
        if not weights_loaded:
            self._autoencoder.fit(point_data.values, point_data.values, epochs=nepoch, batch_size=256, shuffle=True)

    def autoencoder_preprocess(self, **kwargs ):
        niter: int = kwargs.get( 'niter', self.reduce_niter )
        method: str = kwargs.get( 'method', self.reduce_method )
        dropout: float = kwargs.get('dropout', self.reduce_dropout)
        lr = kwargs.get('lr', self.reduce_learning_rate )
        self.vae = (method.strip().lower() == 'vae')
        self.build_encoder( dropout=dropout, lr=lr, **kwargs )
        weights_loaded = self.load_weights(**kwargs)
        initial_epoch = 0
        if not weights_loaded:
            for iter in range(niter):
                initial_epoch = self.general_training( initial_epoch, **kwargs )
                initial_epoch = self.focused_training( initial_epoch, **kwargs )
            aefiles = self.autoencoder_files(**kwargs)
            if self.refresh_model:
                for aef in aefiles:
                    for ifile in glob.glob(aef + ".*"): os.remove(ifile)
            self._autoencoder.save_weights( aefiles[0] )
            self._encoder.save_weights( aefiles[1] )
            lgm().log(f"autoencoder_preprocess completed, saved model weights to files={aefiles}", print=True)

    def general_training(self, initial_epoch = 0, **kwargs ):
        nepoch: int = kwargs.get( 'nepoch', self.reduce_nepoch )
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.tile import Block, Tile
        num_reduce_images = min( dm().modal.num_images, self.reduce_nimages )
        for image_index in range( num_reduce_images ):
            dm().modal.set_current_image(image_index)
            blocks: List[Block] = tm().tile.getBlocks()
            num_training_blocks = min( self.reduce_nblocks, len(blocks) )
            lgm().log(f"Autoencoder general training: {num_training_blocks} blocks for image[{image_index}/{num_reduce_images}]: {dm().modal.image_name}", print=True)
            for iB, block in enumerate(blocks):
                if iB < self.reduce_nblocks:
                    t0 = time.time()
                    point_data, grid = block.getPointData()
                    if point_data.shape[0] > 0:
                        lgm().log( f" ** ITER[{iter}]: Processing block{block.block_coords}, data shape = {point_data.shape}", print=True)
                        history: tf.keras.callbacks.History = self._autoencoder.fit(point_data.data, point_data.data, initial_epoch=initial_epoch,
                                                                 epochs=initial_epoch + nepoch, batch_size=256, shuffle=True)
                        initial_epoch = initial_epoch + nepoch
                        lgm().log(f" Trained autoencoder in {time.time() - t0} sec", print=True)
                    block.initialize()
        return initial_epoch

    def focused_training(self, initial_epoch = 0, **kwargs) -> bool:
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.tile import Block, Tile
        nepoch: int = kwargs.get( 'nepoch', self.reduce_focus_nepoch )
        anom_focus: float = kwargs.get( 'anom_focus', self.reduce_anom_focus )
        if (anom_focus == 0.0) or (nepoch==0): return False

        anomalies = {}
        num_reduce_images = min(dm().modal.num_images, self.reduce_nimages)
        for image_index in range(num_reduce_images):
            dm().modal.set_current_image(image_index)
            blocks: List[Block] = tm().tile.getBlocks()
            num_training_blocks = min(self.reduce_nblocks, len(blocks))
            lgm().log(f"Autoencoder focussed training: {num_training_blocks} blocks for image[{image_index}/{num_reduce_images}]: {dm().modal.image_name}", print=True)
            for iB, block in enumerate(blocks):
                if iB < self.reduce_nblocks:
                    point_data, grid = block.getPointData()
                    if point_data.shape[0] > 0:
                        reproduced_data: np.ndarray = self._autoencoder.predict( point_data.values )
                        anomalies[(image_index,iB)] = self.get_anomaly( point_data.data, reproduced_data )
        full_anomaly: np.ndarray = np.concatenate( list(anomalies.values()) )
        t = self.get_anomaly_threshold(full_anomaly, anom_focus)
        lgm().log(f"autoencoder focus({anom_focus}) training: anomaly threshold = {t}", print=True)
        focused_datsets = []
        for image_index in range(num_reduce_images):
            dm().modal.set_current_image(image_index)
            blocks: List[Block] = tm().tile.getBlocks()
            for iB, block in enumerate(blocks):
                if iB < self.reduce_nblocks:
                    point_data, grid = block.getPointData()
                    if point_data.shape[0] > 0:
                        anomaly = anomalies[(image_index,iB)]
                        focused_point_data = self.get_focused_dataset(point_data.data, anomaly, t )
                        focused_datsets.append( focused_point_data )
                        ntrainsamples = nsamples( focused_datsets )
                        lgm().log(f" --> BLOCK[{image_index}:{block.block_coords}]: ntrainsamples = {ntrainsamples}", print=True)
                        if ntrainsamples > point_data.shape[0]:
                            focused_training_data = np.concatenate( focused_datsets )
                            lgm().log( f" --> Focused Training with #samples = {ntrainsamples}", print=True)
                            history: tf.keras.callbacks.History = self._autoencoder.fit( focused_training_data, focused_training_data, initial_epoch=initial_epoch,
                                                                      epochs=initial_epoch + nepoch, batch_size=256, shuffle=True)
                            initial_epoch = initial_epoch + nepoch
                            focused_datsets = []
        ntrainsamples = nsamples( focused_datsets )
        if ntrainsamples > 0:
            focused_training_data = np.concatenate( focused_datsets )
            lgm().log(f" --> Focused Training with #samples = {ntrainsamples}", print=True)
            history: tf.keras.callbacks.History = self._autoencoder.fit(focused_training_data, focused_training_data, initial_epoch=initial_epoch,
                                                     epochs=initial_epoch + nepoch, batch_size=256, shuffle=True)
        return initial_epoch

    def get_focused_dataset(self, train_data: np.ndarray, anomaly: np.ndarray, threshold: float ) -> np.ndarray:
        rng = np.random.default_rng()
        amask: np.ndarray = (anomaly > threshold)
        anom_data, std_data = train_data[amask], train_data[~amask]
        num_standard_samples = round( anom_data.shape[0]/self.reduce_focus_ratio )
        if num_standard_samples >= std_data.shape[0]:
            return train_data
        else:
            std_data_sample = rng.choice( std_data, num_standard_samples, replace=False, axis=0, shuffle=False )
            new_data = np.concatenate((anom_data, std_data_sample), axis=0)
            return new_data

    def get_anomaly(self, train_data: np.ndarray, reproduced_data ) -> np.ndarray:
        return np.abs(train_data - reproduced_data).sum(axis=-1, keepdims=False)

    def get_anomaly_threshold(self, anomaly: np.ndarray, anom_focus=0.10 ) -> float:
        hist, edges = np.histogram(anomaly, 64)
        counts: np.ndarray = np.cumsum(hist)
        ti: int = np.abs(counts - counts[-1] * (1 - anom_focus)).argmin()
        return edges[ti + 1]

    def autoencoder_reduction(self, train_input: xa.DataArray, **kwargs ) -> Tuple[np.ndarray, np.ndarray]:
        ispecs: List[np.ndarray] = [train_input.data.max(0), train_input.data.min(0), train_input.data.mean(0), train_input.data.std(0)]
        lgm().log(f" autoencoder_reduction: train_input shape = {train_input.shape} ")
        lgm().log(f"   ----> max = { ispecs[0][:64].tolist() } ")
        lgm().log(f"   ----> min = { ispecs[1][:64].tolist() } ")
        lgm().log(f"   ----> ave = { ispecs[2][:64].tolist() } ")
        lgm().log(f"   ----> std = { ispecs[3][:64].tolist() } ")
        self.get_trained_network( **kwargs )
        encoder_result = self._encoder.predict( train_input.values )
        encoded_data: np.ndarray = encoder_result[0] if isinstance(encoder_result, (list, tuple)) else encoder_result
        reproduced_data: np.ndarray = self._autoencoder.predict( train_input.values )
        lgm().log(f" Autoencoder_reduction, result shape = {encoded_data.shape}")
        lgm().log(f" ----> encoder_input: shape = {train_input.shape}, val[5][5] = {train_input.values[:5][:5]} ")
        lgm().log(f" ----> reproduction: shape = {reproduced_data.shape}, val[5][5] = {reproduced_data[:5][:5]} ")
        lgm().log(f" ----> encoding: shape = {encoded_data.shape}, val[5][5] = {encoded_data[:5][:5]}, std = {encoded_data.std(0)} ")
        anomaly = np.abs( train_input.values - reproduced_data ).sum( axis=-1, keepdims=False )
        dmask = anomaly > 0.0
        lgm().log( f" ----> ANOMALY: shape = {anomaly.shape}, range = [{anomaly.min(where=dmask,initial=np.inf)},{anomaly.max()}] ")
        return (encoded_data, reproduced_data)

    @property
    def image_names(self) -> List[str]:
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
            self._image_names = [ self.extract_image_name( image_path ) for image_path in image_path_list ]
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
            msg = f"Setting active image[{self._active_image}]: {self.image_name}"
            lgm().log( msg ); ufm().show( msg )
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
        return self.image_names[self._active_image]

    def get_image_name( self, image_index: int ):
        return self.image_names[ image_index ]

    @property
    def file_selector(self):
        if self._file_selector is None:
            lgm().log( f"Creating file_selector, options={self.image_names}, value={self.image_names[0]}")
            self._file_selector =  ip.Select( options=self.image_names, value=self.image_names[0], layout=ipw.Layout(width='600px') )
        return self._file_selector

    def set_file_selection_observer( self, observer ):
        self.file_selector.observe( observer, names=['value'] )
        return self.file_selector

    def on_image_change( self, event: Dict ):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.spatial.map import MapManager, mm
        self.set_current_image( self.file_selector.index )
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

    def process_block(self, block, has_metadata: bool  ) -> xa.Dataset:
        raise NotImplementedError()

    def reduce(self, train_data: xa.DataArray, **kwargs ) -> Tuple[xa.DataArray,xa.DataArray]:
        reduction_method: int = kwargs.pop( 'method', self.reduce_method )
        with xa.set_options(keep_attrs=True):
            redm = str(reduction_method).lower()
            if redm in [ "autoencoder", "aec", "ae", "vae" ]:
                (reduced_spectra, reproduction) =  self.autoencoder_reduction( train_data, vae=(redm=="vae"), **kwargs )
            elif redm in [ "pca", "ica" ]:
                (reduced_spectra, reproduction) =  self.ca_reduction( train_data, **kwargs )
            else: return  ( train_data, train_data )
            coords = dict( samples=train_data.coords['samples'], band=np.arange( self.model_dims )  )
            reduced_array = xa.DataArray( reduced_spectra, dims=['samples', 'band'], coords=coords )
            reproduced_array = train_data.copy( data=reproduction )
            return (reduced_array, reproduced_array)

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
        return dvars

    @exception_handled
    def loadDataset(self, **kwargs) -> Optional[ Dict[str,Union[xa.DataArray,List,Dict]] ]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.spatial.tile.tile import Block
        from spectraclass.data.base import DataManager, dm, DataType
        lgm().log(f"Load dataset, current = {list(self.datasets.keys())}")
        if self.dsid() not in self.datasets:
            lgm().log(f"Load dataset {self.dsid()}, current datasets = {list(self.datasets.keys())}")
            xdataset: Optional[xa.Dataset] = None
            if not dm().refresh_data:
                xdataset = self.loadDataFile(**kwargs)
            block: Block = tm().getBlock()
            if xdataset is None:
                has_metadata = (self.metadata is not None)
                xdataset = self.process_block( block, has_metadata )
            if (xdataset is None) or (len(xdataset.variables.keys()) == 0):
                lgm().log(f"Warning: Attempt to Load empty dataset {self.dataFile( **kwargs )}")
                return None
            else:
                lgm().log(f" ---> Opening Dataset {self.dsid()}")
                dvars: Dict[str,Union[xa.DataArray,List,Dict]] = self.dset_subsample( xdataset, dsid=self.dsid(), **kwargs )
                attrs = xdataset.attrs.copy()
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
        ufm().show( f" Loading Tile {tm().block_index} " )
        dFile = self.dataFile( **kwargs )
        dataset: Optional[xa.Dataset] = None
        if path.isfile( dFile ):
            dataset = xa.open_dataset( dFile, concat_characters=True )
            dataset.attrs['data_file'] = dFile
            vars = [ f"{vid}{var.dims}" for (vid,var) in dataset.variables.items()]
            coords = [f"{cid}{coord.shape}" for (cid, coord) in dataset.coords.items()]
            lgm().log( f"#GID: loadDataFile: {dFile}, coords={coords}, vars={vars}" )
            lgm().log( f"#GID:  --> coords={coords}")
            lgm().log( f"#GID:  --> vars={vars}")
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

    def readSpectralData(self, **kwargs) -> xa.DataArray:
        raise NotImplementedError( "Attempt to call abstract method 'readSpectralData'")




