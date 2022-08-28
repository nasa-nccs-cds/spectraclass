from ..graph.manager import ActivationFlowManager
from sklearn.decomposition import PCA, FastICA
import tensorflow as tf
keras = tf.keras
from keras import losses, regularizers
from keras.saving.saved_model.load import load as keras_load_model
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from keras.regularizers import l2
import xarray as xa
from ..model.labels import LabelsManager
import traitlets as tl
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable
import pickle, random, time, numpy as np
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Optional, Dict, Tuple
import os, shutil

def pnorm(data: xa.DataArray, dim: int = 1) -> xa.DataArray:
    dave, dmag = np.nanmean(data.values, keepdims=True, axis=dim), np.nanstd(data.values, keepdims=True,                                                                             axis=dim)
    normed_data = (data.values - dave) / dmag
    return data.copy(data=normed_data)

def norm( x: xa.DataArray, axis = 0 ) -> xa.DataArray:
    return ( x - x.data.mean(axis=axis) ) / x.data.std(axis=axis)

def scale( x: xa.DataArray, axis = 0 ) -> xa.DataArray:
    result = x / x.mean(axis=axis)
    result.attrs.update( x.attrs )
    return result

def vae_loss( inputs, outputs, n_features, z_mean, z_log ):
    """ Loss = Recreation loss + Kullback-Leibler loss
    for probability function divergence (ELBO).
    gamma > 1 and capacity != 0 for beta-VAE
    """
    gamma = 1.0
    capacity = 0.0
    reconstruction_loss = mse( inputs, outputs )
    reconstruction_loss *= n_features
    kl_loss = 1 + z_log - K.square(z_mean) - K.exp(z_log)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    kl_loss = gamma * K.abs( kl_loss - capacity )
    return K.mean(reconstruction_loss + kl_loss)

class ReductionManager(SCSingletonConfigurable):
    init = tl.Unicode("random").tag(config=True,sync=True)
    loss = tl.Unicode("mean_squared_error").tag(config=True,sync=True)
    # Losses: mean_squared_error, cosine_similarity, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error
    # See: https://keras.io/api/losses/regression_losses/
    nepochs = tl.Int( 200 ).tag(config=True,sync=True)
    alpha = tl.Float( 0.9 ).tag(config=True,sync=True)
    ndim = tl.Int( 3 ).tag(config=True,sync=True)
    target_weight = tl.Float( 0.5 ).tag(config=True,sync=True)

    UNDEF = -1
    INIT = 0
    NEW_DATA = 1
    PROCESSED = 2

    def __init__(self, **kwargs):
        super(ReductionManager, self).__init__(**kwargs)
        self._mapper = {}
        self._dsid = None
        self.conf = kwargs
        self.ndim = 3
        self._state = self.UNDEF
        self._samples_coord = None
        self._autoencoder = None
        self._encoder = None
        self.vae = None

    @exception_handled
    def reduce(self, train_data: xa.DataArray, reduction_method: Optional[str], modelkey: str, ndim: int, **kwargs ) -> Tuple[np.ndarray, xa.DataArray]:
        with xa.set_options(keep_attrs=True):
            redm = str(reduction_method).lower()
            if redm in [ "autoencoder", "aec", "ae", "vae" ]:
                return self.autoencoder_reduction( train_data, ndim, modelkey, vae=(redm=="vae"), **kwargs )
            elif redm in [ "pca", "ica" ]:
                return self.ca_reduction( train_data, ndim, redm )
            else: return  (train_data.data,train_data)

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

    def ca_reduction(self, train_input: xa.DataArray, ndim: int, method = "pca" ) -> Tuple[np.ndarray, xa.DataArray]:
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
            reproduction: xa.DataArray = train_input.copy( data = mapper.inverse_transform(reduced_features) )
            return ( reduced_features, reproduction )
        else:
            normed_input = norm( train_input )
            reduced_features: np.ndarray = mapper.transform( normed_input.data )
            reproduction: xa.DataArray = train_input.copy( data = mapper.inverse_transform(reduced_features) )
            return (reduced_features, reproduction )

    def _load_network(self, model_dims: int, key: str  ) -> bool:
        aefiles = self.autoencoder_files( model_dims, key )
        if not os.path.exists( aefiles[0] ):
            lgm().log( f"AEC model file does not exist: {aefiles[0]}")
            return False
        else:
            lgm().log( f"#AEC: LOADING ENCODER from '{aefiles[0]}'")
            self._autoencoder = keras_load_model(  f"{aefiles[0]}" )
            self._encoder =     keras_load_model(  f"{aefiles[1]}" )
            return True

    def get_trained_network(self, model_dims: int, key: str, **kwargs ):
        self.vae = kwargs.pop( 'vae', True )
        if self._autoencoder is None:
            if not self._load_network( model_dims, key ):
                self.autoencoder_preprocess( model_dims, key, **kwargs )

    def build_network(self, input_dims: int, model_dims: int, **kwargs):
        if self.vae:
            self._build_vae_model(input_dims, model_dims, **kwargs)
        else:
            self._build_ae_model(input_dims, model_dims, **kwargs)

    def _build_ae_model(self, input_dims: int, model_dims: int, **kwargs):
        lgm().log(f"#AEC: RM BUILD AEC NETWORK: {input_dims} -> {model_dims}")
        sparsity: float = kwargs.get( 'sparsity', 0.0 )
        reduction_factor = 2
        inputlayer = Input(shape=[input_dims])
        activation = 'tanh'
        optimizer = 'rmsprop'
        layer_dims, x = int(round(input_dims / reduction_factor)), inputlayer
        while layer_dims > model_dims:
            x = Dense(layer_dims, activation=activation)(x)
            layer_dims = int(round(layer_dims / reduction_factor))
        encoded = x = Dense(model_dims, activation=activation, activity_regularizer=regularizers.l1(sparsity))(x)
        layer_dims = int(round(model_dims * reduction_factor))
        while layer_dims < input_dims:
            x = Dense(layer_dims, activation=activation)(x)
            layer_dims = int(round(layer_dims * reduction_factor))
        decoded = Dense(input_dims, activation='linear')(x)
        #        modelcheckpoint = ModelCheckpoint('xray_auto.weights', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        #        earlystopping = EarlyStopping(monitor='loss', min_delta=0., patience=100, verbose=1, mode='auto')
        self._autoencoder = Model(inputs=[inputlayer], outputs=[decoded])
        self._encoder = Model(inputs=[inputlayer], outputs=[encoded])
        #        autoencoder.summary()
        #        encoder.summary()
        self._autoencoder.compile( loss=self.loss, optimizer=optimizer )
        self.log = lgm().log(f" BUILD Autoencoder network: input_dims = {input_dims} ")


    def sampling(self, args):
        import tensorflow as tf
        keras = tf.keras
        from keras import backend as K
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
        batch = K.shape(z_mean)[0]  # batch size
        dim = K.int_shape(z_mean)[1]  # latent dimension
        epsilon = K.random_normal(shape=(batch, dim))  # mean=0, std=1.0
        return z_mean + K.exp(0.5 * z_log) * epsilon

    def _build_vae_model(self, input_dims: int, model_dims: int, **kwargs ):
        hidden_activation = "relu"
        output_activation = 'linear'
        optimizer = 'rmsprop'  # 'rmsprop'
        lgm().log(f"#AEC: RM BUILD VAE NETWORK: {input_dims} -> {model_dims}")
        reduction_factor = 2.0
        verbose = 1
        dropout_rate = 0.1
        l2_regularizer = 0.1
        inputs = Input(shape=(input_dims,))
        layer = Dense(input_dims, activation=hidden_activation)(inputs)
        layer_dims = int(round(input_dims / reduction_factor))
        while layer_dims > model_dims:
            layer = Dense(layer_dims, activation=hidden_activation, activity_regularizer=l2(l2_regularizer))(layer)
            layer = Dropout(dropout_rate)(layer)
            layer_dims = int(round(layer_dims / reduction_factor))

        z_mean = Dense(model_dims)(layer)
        z_log = Dense(model_dims)(layer)
        z = Lambda(self.sampling, output_shape=(model_dims,))([z_mean, z_log])

        self._encoder = Model(inputs, [z_mean, z_log, z])
        if verbose >= 1: self._encoder.summary()

        latent_inputs = Input(shape=(model_dims,))
        layer = Dense(model_dims, activation=hidden_activation)(latent_inputs)
        layer_dims = int(round(model_dims * reduction_factor))
        while layer_dims < input_dims:
            layer = Dense(layer_dims, activation=hidden_activation)(layer)
            layer = Dropout(dropout_rate)(layer)
            layer_dims = int(round(layer_dims * reduction_factor))

        outputs = Dense(input_dims, activation=output_activation)(layer)

        self._decoder = Model(latent_inputs, outputs)
        if verbose >= 1: self._decoder.summary()

        outputs = self._decoder(self._encoder(inputs)[2])
        self._autoencoder = Model(inputs, outputs)
        self._autoencoder.add_loss( vae_loss( inputs, outputs, input_dims, z_mean, z_log ) )
        self._autoencoder.compile(optimizer=optimizer)
        if verbose >= 1: self._autoencoder.summary()
        self.log = lgm().log(f" BUILD Autoencoder network: input_dims = {input_dims} ")

    def autoencoder_files(self, model_dims: int, key: str ) -> List[str]:
        from spectraclass.data.base import DataManager, dm
        aefiles = [f"{dm().cache_dir}/autoencoder.{model_dims}.{key}", f"{dm().cache_dir}/encoder.{model_dims}.{key}.h5"]
        return aefiles

    def autoencoder_preprocess(self, model_dims: int, key: str, **kwargs ):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        refresh = kwargs.get( 'refresh', False )
        nepoch: int = kwargs.get( 'nepoch', 5 )
        niter: int = kwargs.get( 'niter', 2 )
        method: str = kwargs.pop( 'method', 'vae' )
        self.vae = (method.strip().lower() == 'vae')
        self._autoencoder = None
        aefiles = self.autoencoder_files( model_dims, key )
        if refresh:
            for aedir in aefiles: shutil.rmtree( aedir, ignore_errors=True )
        if not os.path.exists(aefiles[0]):
            ufm().show( f"Initializing autoencoder{key}[{model_dims}]: file = {aefiles[0]}", print=True, log=True )
            for iter in range(niter):
                for image_index in range(dm().modal.num_images):
                    dm().modal.set_current_image(image_index)
                    lgm().log(f"Preprocessing data blocks{tm().block_dims} for image {dm().modal.image_name}", print=True)
                    for block in tm().tile.getBlocks():
                        t0 = time.time()
                        point_data, grid = block.getPointData()
                        if point_data.shape[0] > 0:
                            norm_data = pnorm(point_data)
                            lgm().log(f"\nITER[{iter}]: Processing block{block.block_coords}, data shape = {point_data.shape}", print=True )
                            if self._autoencoder is None:
                                self.build_network( norm_data.shape[1], model_dims)
                            self._autoencoder.fit(norm_data.data, norm_data.data, epochs=nepoch, batch_size=256, shuffle=True)
                            lgm().log(f" Trained autoencoder in {time.time() - t0} sec", print=True)
                        block.initialize()
            self._autoencoder.save(aefiles[0])
            self._encoder.save(aefiles[1])
            lgm().log(f"autoencoder_preprocess completed, saved model to {aefiles}", print=True)
            ufm().show(f"autoencoder init complete", print=True)

    def autoencoder_reduction(self, train_input: xa.DataArray, model_dims: int, model_key: str, **kwargs ) -> Tuple[np.ndarray, xa.DataArray]:
        ispecs: List[np.ndarray] = [train_input.data.max(0), train_input.data.min(0), train_input.data.mean(0), train_input.data.std(0)]
        lgm().log(f" autoencoder_reduction: train_input shape = {train_input.shape}, model dims={model_dims}, model key={model_key} ")
        lgm().log(f"   ----> max = { ispecs[0][:64].tolist() } ")
        lgm().log(f"   ----> min = { ispecs[1][:64].tolist() } ")
        lgm().log(f"   ----> ave = { ispecs[2][:64].tolist() } ")
        lgm().log(f"   ----> std = { ispecs[3][:64].tolist() } ")
        self.get_trained_network( model_dims, model_key, **kwargs )
        encoder_result = self._encoder.predict( train_input.data )
        encoded_data: np.ndarray = encoder_result[0] if isinstance(encoder_result, (list, tuple)) else encoder_result
        reproduced_data: np.ndarray = self._autoencoder.predict( train_input.data )
        reproduction: xa.DataArray = train_input.copy( data=reproduced_data )
        lgm().log(f" Autoencoder_reduction, result shape = {encoded_data.shape}")
        lgm().log(f" ----> encoder_input: shape = {train_input.shape}, val[5][5] = {train_input.data[:5][:5]} ")
        lgm().log(f" ----> reproduction: shape = {reproduced_data.shape}, val[5][5] = {reproduced_data[:5][:5]} ")
        lgm().log(f" ----> encoding: shape = {encoded_data.shape}, val[5][5] = {encoded_data[:5][:5]}, std = {encoded_data.std(0)} ")
        return (encoded_data, reproduction)

    def umap_init( self,  point_data: xa.DataArray, **kwargs ) -> Optional[xa.DataArray]:
        from .cpu import UMAP
        self._state = self.NEW_DATA
        self._dsid = point_data.attrs['dsid']
        LabelsManager.instance()._init_labels_data(point_data)
        mapper: UMAP = self.getUMapper( self._dsid, self.ndim, True )
        mapper.scoord = point_data.coords['samples']
        mapper.input_data = point_data.values
        if point_data.shape[1] <= self.ndim:
            mapper.set_embedding(mapper.input_data)
            return point_data
        else:
            lgm().log( f"umap_init: init = {self.init}")
            if self.init == "autoencoder":
                [( reduction, reproduction, _ )] = self.autoencoder_reduction( point_data, None, self.ndim, 50 )
                mapper.init_embedding(reduction)
            mapper.init = self.init
            kwargs['nepochs'] = 1
            labels_data: np.ndarray = LabelsManager.instance().getLabelsArray().values
            lgm().log(f"INIT UMAP embedding with input data{point_data.dims}, shape = {mapper.input_data.shape}, labels_data shape = {labels_data.shape}, parms: {kwargs}")
            mapper.embed( mapper.input_data, labels_data, **kwargs )
            ecoords = dict( samples=point_data.samples, model=np.arange(0,self.ndim) )
            return xa.DataArray( mapper.embedding, dims=['samples','model'], coords=ecoords, attrs=point_data.attrs )

    def umap_embedding( self, **kwargs ) -> Optional[np.ndarray]:
        from .cpu import UMAP
        mapper: UMAP = self.getUMapper(self._dsid, self.ndim)
        if 'nepochs' not in kwargs.keys():   kwargs['nepochs'] = self.nepochs
        if 'alpha' not in kwargs.keys():   kwargs['alpha'] = self.alpha
        self._state = self.PROCESSED
        labels_data: np.ndarray = kwargs.get( 'labels', LabelsManager.instance().getLabelsArray()).values
        lgm().log( f"Executing UMAP embedding with input data shape = {mapper.input_data.shape}, parms: {kwargs}")
        labels_data[ labels_data == 0 ] = -1
        mapper.embed( mapper.input_data, labels_data, **kwargs )
        return mapper.embedding

    def xa_umap_embedding( self, **kwargs ) -> Optional[xa.DataArray]:
        from .cpu import UMAP
        mapper: UMAP = self.getUMapper(self._dsid, self.ndim)
        if mapper.embedding is None: self.umap_embedding( **kwargs )
        return None if mapper.embedding is None else self.wrap_embedding( mapper.scoord, mapper.embedding, **kwargs )

    def wrap_embedding(self, ax_samples: xa.DataArray, embedding: np.ndarray, **kwargs )-> xa.DataArray:
        ax_model = np.arange( embedding.shape[1] )
        return xa.DataArray( embedding, dims=['samples','model'], coords=dict( samples=ax_samples, model=ax_model ) )

    def getUMapper(self, dsid: str, ndim: int, refresh=False ):
        mid = f"{ndim}-{dsid}"
        nneighbors = ActivationFlowManager.instance().nneighbors
        mapper = self._mapper.get( mid )
        if refresh or ( mapper is None ):
            from .base import UMAP
            kwargs = dict( n_neighbors=nneighbors, init=self.init, target_weight=self.target_weight, n_components=ndim, **self.conf )
            mapper = UMAP.instance( **kwargs )
            self._mapper[mid] = mapper
        self._current_mapper = mapper
        return mapper

def rm():
    return ReductionManager.instance()
