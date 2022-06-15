from typing import List, Union, Tuple, Dict
from typing import List, Union, Tuple, Optional, Dict
from ..graph.manager import ActivationFlowManager
from sklearn.decomposition import PCA, FastICA
from tensorflow.keras.models import Model
from keras.models import load_model
import xarray as xa
import numpy as np, time, traceback
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable

def norm( x: xa.DataArray, axis = 0 ) -> xa.DataArray:
    return ( x - x.data.mean(axis=axis) ) / x.data.std(axis=axis)

def scale( x: xa.DataArray, axis = 0 ) -> xa.DataArray:
    result = x / x.mean(axis=axis)
    result.attrs.update( x.attrs )
    return result

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

    @exception_handled
    def reduce(self, train_data: xa.DataArray, test_data: List[xa.DataArray], reduction_method: str, ndim: int, nepochs: int = 100, **kwargs ) -> List[Tuple[np.ndarray, xa.DataArray, xa.DataArray]]:
        with xa.set_options(keep_attrs=True):
            if test_data is None: test_data = [train_data]
            redm = reduction_method.lower()
            if redm in [ "autoencoder", "aec", "ae" ]:
                return self.autoencoder_reduction( train_data, test_data, ndim, nepochs, **kwargs )
            elif redm in [ "pca", "ica" ]:
                return self.ca_reduction( train_data, test_data, ndim, redm )
            else: return [ (td.data,td,td) for td in test_data ]

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

    def ca_reduction(self, train_input: xa.DataArray, test_inputs: Optional[List[xa.DataArray]], ndim: int, method = "pca" ) -> List[Tuple[np.ndarray, xa.DataArray, xa.DataArray]]:
        if method == "pca":
            mapper = PCA(n_components=ndim)
        elif method == "ica":
            mapper = FastICA(n_components=ndim)
        else: raise Exception( f"Unknown reduction methos: {method}")
        normed_train_input: xa.DataArray = norm( train_input  )
        mapper.fit( normed_train_input.data )
        if method == "pca":
            lgm().log( f"PCA reduction[{ndim}], Percent variance explained: {mapper.explained_variance_ratio_ * 100}" )
        if test_inputs is None:
            reduced_features: np.ndarray = mapper.transform( normed_train_input.data )
            reproduction: xa.DataArray = train_input.copy( data = mapper.inverse_transform(reduced_features) )
            return [ ( reduced_features, reproduction, normed_train_input ) ]
        else:
            results = []
            for iT, test_input in enumerate(test_inputs):
                normed_input = norm( test_input )
                reduced_features: np.ndarray = mapper.transform( normed_input.data )
                reproduction: xa.DataArray = test_input.copy( data = mapper.inverse_transform(reduced_features) )
                results.append( (reduced_features, reproduction, normed_input ) )
            return results

    def _load_network(self, key: str, model_dims: int, **kwargs ):
        from spectraclass.data.base import DataManager, dm
        lgm().log( f"#AEC: LOADING ENCODER from '{dm().cache_dir}/encoder.{model_dims}.{key}'")
        self._autoencoder = load_model(  f"{dm().cache_dir}/autoencoder.{model_dims}.{key}" )
        self._encoder =     load_model(  f"{dm().cache_dir}/encoder.{model_dims}.{key}" )

    def get_network( self, input_dims: int, model_dims: int, **kwargs ):
        refresh: bool = kwargs.get('refresh', True )
        key: str = kwargs.get( 'modelkey', "" )
        if (self._autoencoder is None) or refresh:
            if key:     self._load_network( key, model_dims, **kwargs )
            else:       self._build_network( input_dims, model_dims, **kwargs )
        return self._autoencoder, self._encoder, bool(key)

    def _build_network( self, input_dims: int, model_dims: int, **kwargs  ):
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.models import Model
        from tensorflow.keras import losses, regularizers
        lgm().log(f"#AEC: BUILD AEC NETWORK")
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

    def autoencoder_reduction(self, train_input: xa.DataArray, test_inputs: Optional[List[xa.DataArray]], model_dims: int, epochs: int = 100, **kwargs ) -> List[Tuple[np.ndarray, xa.DataArray, xa.DataArray]]:
        from tensorflow.keras.models import Model
        autoencoder: Model = None
        ispecs: List[np.ndarray] = [train_input.data.max(0), train_input.data.min(0), train_input.data.mean(0), train_input.data.std(0)]
        lgm().log(f" autoencoder_reduction: train_input shape = {train_input.shape} ")
        lgm().log(f"   ----> max = { ispecs[0][:64].tolist() } ")
        lgm().log(f"   ----> min = { ispecs[1][:64].tolist() } ")
        lgm().log(f"   ----> ave = { ispecs[2][:64].tolist() } ")
        lgm().log(f"   ----> std = { ispecs[3][:64].tolist() } ")

#        lgm().log(f"   ----> #NaN[axis=0] = { nan_counts[:200] }" )
        results = []
        if test_inputs is None: test_inputs = [ train_input ]
        for iT, test_input in enumerate(test_inputs):
            try:
                autoencoder, encoder, prebuilt = self.get_network( train_input.shape[1], model_dims, **kwargs )
                if not prebuilt:
                    lgm().log(f"#AEC: TRAIN AEC NETWORK")
                    autoencoder.fit( train_input.data, train_input.data, epochs=epochs, batch_size=256, shuffle=True )
                encoded_data: np.ndarray = encoder.predict( test_input.data )
                reproduced_data: np.ndarray = autoencoder.predict( test_input.data )
                reproduction: xa.DataArray = test_input.copy( data=reproduced_data )
                results.append( (encoded_data, reproduction, test_input ) )
                if iT == 0:
                    lgm().log(f" Autoencoder_reduction, result shape = {encoded_data.shape}")
                    lgm().log(f" ----> encoder_input: shape = {test_input.shape}, val[5][5] = {test_input.data[:5][:5]} ")
                    lgm().log(f" ----> reproduction: shape = {reproduced_data.shape}, val[5][5] = {reproduced_data[:5][:5]} ")
                    lgm().log(f" ----> encoding: shape = {encoded_data.shape}, val[5][5] = {encoded_data[:5][:5]}, std = {encoded_data.std(0)} ")
            except Exception as err:
                lgm().exception( f"Unable to process test input[{iT}], input shape = {test_input.shape}, error = {err}" )
        return results

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
