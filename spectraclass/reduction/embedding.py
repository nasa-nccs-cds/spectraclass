from typing import List, Union, Tuple, Dict
from typing import List, Union, Tuple, Optional, Dict
from ..graph.manager import ActivationFlowManager
import xarray as xa
import numpy as np, time, traceback
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
from spectraclass.model.base import AstroConfigurable

class ReductionManager(tlc.SingletonConfigurable, AstroConfigurable):
    init = tl.Unicode("spectral").tag(config=True,sync=True)
    nepochs = tl.Int( 100 ).tag(config=True,sync=True)
    nneighbors = tl.Int(15).tag(config=True, sync=True)
    alpha = tl.Float( 0.25 ).tag(config=True,sync=True)
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

    def reduce(self, inputs: np.ndarray, reduction_method: str, ndim: int, nepochs: int = 1 ) -> np.ndarray:
        if reduction_method.lower() == "autoencoder": return self.autoencoder_reduction( inputs, ndim, nepochs )

    def xreduce(self, inputs: xa.DataArray, reduction_method: str, ndim: int ) -> xa.DataArray:
        if reduction_method.lower() == "autoencoder":
            encoded_data = self.autoencoder_reduction( inputs.values, ndim )
            coords = {inputs.dims[0]: inputs.coords[inputs.dims[0]], inputs.dims[1]: np.arange(ndim)}
            return xa.DataArray(encoded_data, dims=inputs.dims, coords=coords, attrs=inputs.attrs)
        return inputs

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

    def autoencoder_reduction( self, encoder_input: np.ndarray, ndim: int, epochs: int = 1 ) -> np.ndarray:
        from keras.layers import Input, Dense
        from keras.models import Model
        input_dims = encoder_input.shape[1]
        reduction_factor = 1.7
        inputlayer = Input( shape=[input_dims] )
        activation = 'tanh'
        encoded = None
        layer_dims, x = input_dims, inputlayer
        while layer_dims > ndim:
            x = Dense(layer_dims, activation=activation)(x)
            layer_dims = int( round( layer_dims / reduction_factor ))
        layer_dims = ndim
        while layer_dims < input_dims:
            x = Dense(layer_dims, activation=activation)(x)
            if encoded is None: encoded = x
            layer_dims = int( round( layer_dims * reduction_factor ))
        decoded = Dense( input_dims, activation='sigmoid' )(x)

#        modelcheckpoint = ModelCheckpoint('xray_auto.weights', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
#        earlystopping = EarlyStopping(monitor='loss', min_delta=0., patience=100, verbose=1, mode='auto')
        autoencoder = Model(inputs=[inputlayer], outputs=[decoded])
        encoder = Model(inputs=[inputlayer], outputs=[encoded])
        autoencoder.compile(loss='mse', optimizer='rmsprop')
        autoencoder.fit( encoder_input, encoder_input, epochs=epochs, batch_size=256, shuffle=True )
        return  encoder.predict( encoder_input )

    def umap_init( self,  point_data: xa.DataArray, **kwargs ) -> Optional[np.ndarray]:
        from .cpu import UMAP
        self._state = self.NEW_DATA
        self._dsid = point_data.attrs['dsid']
        LabelsManager.instance().initLabelsData(point_data)
        mapper: UMAP = self.getUMapper(self._dsid, self.ndim)
        mapper.scoord = point_data.coords['samples']
        mapper.input_data = point_data.values
        mapper.flow = ActivationFlowManager.instance().getActivationFlow(point_data)
        if point_data.shape[1] <= self.ndim:
            mapper.set_embedding(mapper.input_data)
        else:
            print( f"umap_init: init = {self.init}")
            mapper.init = self.init
            kwargs['nepochs'] = 1
            labels_data: np.ndarray = LabelsManager.instance().labels_data().values
            mapper.embed( mapper.input_data, labels_data, nngraph= mapper.flow.nnd, **kwargs)
        return mapper.embedding

    def umap_embedding( self, **kwargs ) -> Optional[np.ndarray]:
        from .cpu import UMAP
        mapper: UMAP = self.getUMapper(self._dsid, self.ndim)
        if 'nepochs' not in kwargs.keys():   kwargs['nepochs'] = self.nepochs
        if 'alpha' not in kwargs.keys():   kwargs['alpha'] = self.alpha
        self._state = self.PROCESSED
        labels_data: np.ndarray = LabelsManager.instance().labels_data().values
        mapper.clear_initialization()
        mapper.init = mapper.embedding
        mapper.embed( mapper.input_data, labels_data, nngraph= mapper.flow.nnd, **kwargs )
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
