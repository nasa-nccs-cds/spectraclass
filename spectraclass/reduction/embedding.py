from typing import List, Union, Tuple, Dict
from typing import List, Union, Tuple, Optional, Dict
from ..graph.manager import ActivationFlowManager
import xarray as xa
import numpy as np, time, traceback
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
from spectraclass.model.base import SCSingletonConfigurable

class ReductionManager(SCSingletonConfigurable):
    init = tl.Unicode("random").tag(config=True,sync=True)
    nepochs = tl.Int( 200 ).tag(config=True,sync=True)
    nneighbors = tl.Int( 8 ).tag(config=True, sync=True)
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

    def reduce(self, inputs: np.ndarray, reduction_method: str, ndim: int, nepochs: int = 100, sparsity: float = 0.0 ) -> Tuple[np.ndarray,np.ndarray]:
        if reduction_method.lower() == "autoencoder": return self.autoencoder_reduction( inputs, ndim, nepochs, sparsity )
        else: return ( inputs, inputs )

    def xreduce(self, inputs: xa.DataArray, reduction_method: str, ndim: int ) -> Tuple[xa.DataArray,xa.DataArray]:
        if reduction_method.lower() == "autoencoder":
            ( encoded_data, reproduced_data ) = self.autoencoder_reduction( inputs.values, ndim )
            coords = {inputs.dims[0]: inputs.coords[inputs.dims[0]], inputs.dims[1]: np.arange(ndim)}
            x_encoded_data = xa.DataArray(encoded_data, dims=inputs.dims, coords=coords, attrs=inputs.attrs)
            x_reproduced_data = inputs.copy( data=reproduced_data )
            return ( x_encoded_data, x_reproduced_data )
        return ( inputs, inputs )

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

    def autoencoder_reduction( self, encoder_input: np.ndarray, ndim: int, epochs: int = 100, sparsity: float = 0.0 ) -> Tuple[np.ndarray,np.ndarray]:
        from keras.layers import Input, Dense
        from keras.models import Model
        from keras import losses, regularizers
        from scipy.stats import entropy
        input_dims = encoder_input.shape[1]
        reduction_factor = 2
        inputlayer = Input( shape=[input_dims] )
        activation = 'tanh'
        optimizer = 'rmsprop'
        loss = "cosine_similarity"
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

        autoencoder.compile(loss=loss, optimizer=optimizer )
        autoencoder.fit( encoder_input, encoder_input, epochs=epochs, batch_size=256, shuffle=True )
        encoded_data = encoder.predict( encoder_input )
        scaled_encoding = encoded_data/encoded_data.std()
        reproduction = autoencoder.predict( encoder_input )
        print(f" Autoencoder_reduction with sparsity={sparsity}, result: shape = {encoded_data.shape}")
        print(f" ----> encoder_input: shape = {encoder_input.shape}, val[5][5] = {encoder_input[:5][:5]} ")
        print(f" ----> reproduction: shape = {reproduction.shape}, val[5][5] = {reproduction[:5][:5]} ")
        print(f" ----> encoding: shape = {scaled_encoding.shape}, val[5][5]108 = {scaled_encoding[:5][:5]} ")
        return (scaled_encoding, reproduction )

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
            print( f"umap_init: init = {self.init}")
            if self.init == "autoencoder":
                (reduction, reproduction) = self.autoencoder_reduction( point_data.values, self.ndim, 50 )
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
        print( f"Executing UMAP embedding with input data shape = {mapper.input_data.shape}, parms: {kwargs}")
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
