from typing import List, Union, Tuple, Dict
from typing import List, Union, Tuple, Optional, Dict
from ..graph.manager import ActivationFlowManager
import xarray as xa
import numpy as np, time, traceback
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm, exception_handled
from spectraclass.model.base import SCSingletonConfigurable

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

    def refresh(self):
        self._mapper = {}

    def reduce(self, train_data: np.ndarray, test_data: List[np.ndarray], reduction_method: str, ndim: int, nepochs: int = 100, sparsity: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if test_data is None: test_data = [train_data]
        if reduction_method.lower() == "autoencoder": return self.autoencoder_reduction(train_data, test_data, ndim, nepochs, sparsity)
        else: return [ (td,td,td) for td in test_data ]

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

    def autoencoder_reduction(self, train_input: np.ndarray, test_inputs: Optional[List[np.ndarray]], ndim: int, epochs: int = 100, sparsity: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        from keras.layers import Input, Dense
        from keras.models import Model
        from keras import losses, regularizers
        if test_inputs is None: test_inputs = [ train_input ]
        input_dims = train_input.shape[1]
        reduction_factor = 2
        inputlayer = Input( shape=[input_dims] )
        activation = 'tanh'
        optimizer = 'rmsprop'
        layer_dims, x = int( round( input_dims / reduction_factor )), inputlayer
        while layer_dims > ndim:
            x = Dense(layer_dims, activation=activation)(x)
            layer_dims = int( round( layer_dims / reduction_factor ))
        encoded = x = Dense( ndim, activation=activation, activity_regularizer=regularizers.l1( sparsity ) )(x)
        layer_dims = int( round( ndim * reduction_factor ))
        while layer_dims < input_dims:
            x = Dense(layer_dims, activation=activation)(x)
            layer_dims = int( round( layer_dims * reduction_factor ))
        decoded = Dense( input_dims, activation='sigmoid' )(x)

#        modelcheckpoint = ModelCheckpoint('xray_auto.weights', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
#        earlystopping = EarlyStopping(monitor='loss', min_delta=0., patience=100, verbose=1, mode='auto')
        autoencoder = Model(inputs=[inputlayer], outputs=[decoded])
        encoder = Model(inputs=[inputlayer], outputs=[encoded])
        autoencoder.summary()
        encoder.summary()

        autoencoder.compile(loss=self.loss, optimizer=optimizer )
        autoencoder.fit(train_input, train_input, epochs=epochs, batch_size=256, shuffle=True)
        results = []
        for iT, test_input in enumerate(test_inputs):
            try:
                encoded_data = encoder.predict(test_input)
                scaled_encoding = encoded_data/encoded_data.std()
                reproduction = autoencoder.predict(test_input)
                results.append( (scaled_encoding, reproduction, test_input ) )
                if iT == 0:
                    lgm().log(f" Autoencoder_reduction with sparsity={sparsity}, result: shape = {encoded_data.shape}")
                    lgm().log(f" ----> encoder_input: shape = {train_input.shape}, val[5][5] = {train_input[:5][:5]} ")
                    lgm().log(f" ----> reproduction: shape = {reproduction.shape}, val[5][5] = {reproduction[:5][:5]} ")
                    lgm().log(f" ----> encoding: shape = {scaled_encoding.shape}, val[5][5]108 = {scaled_encoding[:5][:5]} ")
            except Exception as err:
                lgm().exception( f"Unable to process test input[{iT}], input shape = {test_input.shape}, error = {err}" )
        return results

    def umap_init( self,  point_data: xa.DataArray, **kwargs ) -> Optional[np.ndarray]:
        from .cpu import UMAP
        self._state = self.NEW_DATA
        self._dsid = point_data.attrs['dsid']
        LabelsManager.instance().initLabelsData(point_data)
        mapper: UMAP = self.getUMapper(self._dsid, self.ndim)
        mapper.scoord = point_data.coords['samples']
        mapper.input_data = point_data.values
        if point_data.shape[1] <= self.ndim:
            mapper.set_embedding(mapper.input_data)
        else:
            lgm().log( f"umap_init: init = {self.init}")
            if self.init == "autoencoder":
                [( reduction, reproduction, _ )] = self.autoencoder_reduction( point_data.values, None, self.ndim, 50 )
                mapper.init_embedding(reduction)
            mapper.init = self.init
            kwargs['nepochs'] = 1
            labels_data: np.ndarray = LabelsManager.instance().labels_data().values
            mapper.embed( mapper.input_data, labels_data, **kwargs)
        return mapper.embedding

    def umap_embedding( self, **kwargs ) -> Optional[np.ndarray]:
        from .cpu import UMAP
        mapper: UMAP = self.getUMapper(self._dsid, self.ndim)
        if 'nepochs' not in kwargs.keys():   kwargs['nepochs'] = self.nepochs
        if 'alpha' not in kwargs.keys():   kwargs['alpha'] = self.alpha
        self._state = self.PROCESSED
        labels_data: xa.DataArray = kwargs.get( 'labels', LabelsManager.instance().labels_data() )
        lgm().log( f"Executing UMAP embedding with input data shape = {mapper.input_data.shape}, parms: {kwargs}")
        mapper.embed( mapper.input_data, labels_data.values, **kwargs )
        return mapper.embedding

    def xa_umap_embedding( self, **kwargs ) -> Optional[xa.DataArray]:
        from .cpu import UMAP
        mapper: UMAP = self.getUMapper(self._dsid, self.ndim)
        if mapper.embedding is None: self.umap_embedding( **kwargs )
        return None if mapper.embedding is None else self.wrap_embedding( mapper.scoord, mapper.embedding, **kwargs )

    def wrap_embedding(self, ax_samples: xa.DataArray, embedding: np.ndarray, **kwargs )-> xa.DataArray:
        ax_model = np.arange( embedding.shape[1] )
        return xa.DataArray( embedding, dims=['samples','model'], coords=dict( samples=ax_samples, model=ax_model ) )

    def getUMapper(self, dsid: str, ndim: int ):
        mid = f"{ndim}-{dsid}"
        nneighbors = ActivationFlowManager.instance().nneighbors
        mapper = self._mapper.get( mid )
        if ( mapper is None ):
            from .base import UMAP
            kwargs = dict( n_neighbors=nneighbors, init=self.init, target_weight=self.target_weight, n_components=ndim, **self.conf )
            mapper = UMAP.instance( **kwargs )
            self._mapper[mid] = mapper
        self._current_mapper = mapper
        return mapper

def rm():
    return ReductionManager.instance()
