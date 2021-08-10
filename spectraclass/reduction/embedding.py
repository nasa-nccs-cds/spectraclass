from typing import List, Union, Tuple, Optional, Dict
from ..graph.manager import ActivationFlowManager
from torch.utils.data import DataLoader
import xarray as xa
import torch
import torch.nn as nn
import numpy as np, time, traceback
from ..model.labels import LabelsManager
from spectraclass.data.loaders import xaTorchDataset, xaTorchDataLoader
import traitlets as tl
from spectraclass.util.logs import LogManager, lgm, exception_handled
from spectraclass.model.base import SCSingletonConfigurable

def norm( x: xa.DataArray, axis = 0 ) -> xa.DataArray:
    return ( x - x.data.mean(axis=axis) ) / x.data.std(axis=axis)

def scale( x: xa.DataArray, axis = 0 ) -> xa.DataArray:
    result = x / x.mean(axis=axis)
    result.attrs.update( x.attrs )
    return result

class Autoencoder(nn.Module):
    def __init__(self, input_dims: int, reduced_dims: int, shrink_factor: float = 2 ):
        super(Autoencoder, self).__init__()
        self.loss = nn.MSELoss()
        self.reduced_dims = reduced_dims
        in_features = input_dims
        model_layers, encoder_layers = [], []
        while True:
            out_features = int( round( in_features / shrink_factor ))
            if out_features <= reduced_dims: break
            encoder_layers += [ nn.Linear(in_features=in_features, out_features=out_features ), nn.Tanh() ]
            in_features = out_features
        encoder_layers +=  [ nn.Linear(in_features=in_features, out_features=reduced_dims ), nn.Tanh() ]
        model_layers += encoder_layers
        in_features = reduced_dims
        while True:
            out_features = int( round( in_features * shrink_factor ) )
            if out_features >= input_dims: break
            model_layers += [ nn.Linear(in_features=in_features, out_features=out_features ), nn.Tanh() ]
            in_features = out_features
        model_layers += [ nn.Linear( in_features=in_features, out_features=input_dims ) ]
        self.encoder_decoder = nn.Sequential( *model_layers )
        self.encoder = nn.Sequential(*encoder_layers)

    def embedding( self, input: xa.DataArray ) -> xa.DataArray:
        x: torch.Tensor = torch.from_numpy( input.values )
        reproduction: torch.Tensor = self.encoder(x)
        dims = [ input.dims[0], "model" ]
        coords = { dims[0]: input.coords[ dims[0] ], "model": range(self.reduced_dims) }
        return xa.DataArray( reproduction.numpy(), coords, dims, f"{input.name}_embedding-{self.reduced_dims}", input.attrs )

    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        reproduction = self.encoder_decoder(x)
        return reproduction

    @classmethod
    def getLoader( cls, input: xa.DataArray, batch_size, shuffle ) -> Optional[xaTorchDataLoader]:
        if input is None: return None
        dataset = xaTorchDataset( input, input)
        return xaTorchDataLoader( dataset, batch_size, shuffle )


class ReductionManager(SCSingletonConfigurable):
    loss = tl.Unicode("mean_squared_error").tag(config=True,sync=True)
    ndim = tl.Int( 3 ).tag(config=True,sync=True)

    def __init__(self, **kwargs):
        super(ReductionManager, self).__init__(**kwargs)
        self._dsid = None
        self.conf = kwargs
        self._samples_coord = None

    @exception_handled
    def reduce(self, input_data: xa.DataArray,  ndim: int, nepochs: int = 100 ) -> Tuple[xa.DataArray, xa.DataArray, xa.DataArray]:
        with xa.set_options(keep_attrs=True):
            return self.autoencoder_reduction( input_data, ndim, nepochs )

    def autoencoder_reduction(self, input_data: xa.DataArray, ndim: int, epochs: int, **kwargs ) -> Tuple[xa.DataArray, xa.DataArray, xa.DataArray]:
        input_dims = input_data.shape[1]
        ispecs: List[np.ndarray] = [input_data.data.max(0), input_data.data.min(0), input_data.data.mean(0), input_data.data.std(0)]
        lgm().log(f" autoencoder_reduction: input_data shape = {input_data.shape} ")
        lgm().log(f"   ----> max = {ispecs[0][:64].tolist()} ")
        lgm().log(f"   ----> min = {ispecs[1][:64].tolist()} ")
        lgm().log(f"   ----> ave = {ispecs[2][:64].tolist()} ")
        lgm().log(f"   ----> std = {ispecs[3][:64].tolist()} ")
        reduction_factor = kwargs.get('reduction_factor', 2 )
        batch_size = kwargs.get('batch_size', 64 )
        learning_rate: float = float( kwargs.get('learning_rate', 0.1 ) )
        shuffle: bool = bool( kwargs.get('shuffle', True ) )
        autoencoder = Autoencoder( input_dims, ndim, reduction_factor )
        dataloader: xaTorchDataLoader = Autoencoder.getLoader( input_data, batch_size, shuffle )

        for t in range(epochs):
            lgm().log(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop( autoencoder, dataloader, learning_rate )

        embedding: xa.DataArray = autoencoder.embedding( input_data )
        reproduction: xa.DataArray = autoencoder.reproduction(input_data)

        return ( embedding, reproduction, input_data )

    def train_loop( self, autoencoder: Autoencoder, dataloader: xaTorchDataLoader, learning_rate: float ):
        optimizer = torch.optim.SGD( autoencoder.parameters(), lr=float(learning_rate) )
        size = len( dataloader )
        for batch, (X, y) in enumerate(dataloader):
            lgm().log(f"  --> Batch: {batch}")
            pred = autoencoder(X)
            loss = self.loss(pred, y)

            lgm().log(f"  --- --> Optimize: loss = {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

class EmbeddingManager(SCSingletonConfigurable):
    init = tl.Unicode("random").tag(config=True,sync=True)
    nepochs = tl.Int( 200 ).tag(config=True,sync=True)
    alpha = tl.Float( 0.9 ).tag(config=True,sync=True)
    target_weight = tl.Float( 0.5 ).tag(config=True,sync=True)

    UNDEF = -1
    INIT = 0
    NEW_DATA = 1
    PROCESSED = 2

    def __init__(self, **kwargs):
        super(EmbeddingManager, self).__init__(**kwargs)
        self._mapper = {}
        self._state = self.UNDEF
        self.ndim = 3

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

def em():
    return EmbeddingManager.instance()
