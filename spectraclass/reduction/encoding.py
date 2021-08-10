from typing import List, Union, Tuple, Optional, Dict
from torch.utils.data import DataLoader
import torch.nn.functional as F
import xarray as xa
import torch
import torch.nn as nn
import numpy as np, time, traceback
from spectraclass.data.loaders import xaTorchDataset, xaTorchDataLoader
import traitlets as tl
from spectraclass.util.logs import LogManager, lgm, exception_handled
from spectraclass.model.base import SCSingletonConfigurable

class Autoencoder(nn.Module):

    def addLayer(self, in_features: int, out_features: int, is_encoder: bool ):
        index = len( self._ops ) // 2
        layer_ops = [ f'lin-{index}', f'nl-{index}' ]
        setattr( self, layer_ops[0], nn.Linear( in_features=in_features, out_features=out_features ) )
        setattr( self, layer_ops[1], nn.ReLU() )
        self._ops += layer_ops
        if is_encoder: self._encoder_layers = len( self._ops )

    def __init__(self):
        super(Autoencoder, self).__init__()
        self._encoder_layers = 0
        self._ops = []

    def build( self, input_dims: int, reduced_dims: int, shrink_factor: float = 2 ):
        self.reduced_dims = reduced_dims
        in_features = input_dims

        while True:
            out_features = int( round( in_features / shrink_factor ))
            if out_features <= reduced_dims: break
            self.addLayer( in_features,  out_features, True )
            in_features = out_features
        self.addLayer( in_features,  reduced_dims, True )
        in_features = reduced_dims
        while True:
            out_features = int( round( in_features * shrink_factor ) )
            if out_features >= input_dims: break
            self.addLayer( in_features,  out_features, False )
            in_features = out_features
        self.addLayer( in_features,  input_dims, False )

    def embedding( self, input: xa.DataArray ) -> xa.DataArray:
        x: torch.Tensor = torch.from_numpy( input.values )
        for lid in self._ops[:self._encoder_layers]:
            layer = getattr(self, lid)
            x = layer(x)
        dims = [ input.dims[0], "model" ]
        coords = { dims[0]: input.coords[ dims[0] ], "model": range(self.reduced_dims) }
        return xa.DataArray( x.numpy(), coords, dims, f"{input.name}_embedding-{self.reduced_dims}", input.attrs )

    def forward(self, x: torch.Tensor ) -> torch.Tensor:
        print( "forward ---->")
        for lid in self._ops:
            layer = getattr( self, lid )
            x = layer(x)
        return x

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
        lgm().log(f"   ----> max = {ispecs[0][:64].tolist()} " )
        lgm().log(f"   ----> min = {ispecs[1][:64].tolist()} " )
        lgm().log(f"   ----> ave = {ispecs[2][:64].tolist()} " )
        lgm().log(f"   ----> std = {ispecs[3][:64].tolist()} " )
        reduction_factor = kwargs.get('reduction_factor', 2 )
        batch_size = kwargs.get('batch_size', 64 )
        learning_rate: float = float( kwargs.get('learning_rate', 0.1 ) )
        shuffle: bool = bool( kwargs.get('shuffle', True ) )
        autoencoder = Autoencoder( )
        autoencoder.build( input_dims, ndim, reduction_factor )
        dataloader: xaTorchDataLoader = Autoencoder.getLoader( input_data, batch_size, shuffle )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(autoencoder.parameters(), lr=float(learning_rate))

        for t in range(epochs):
            lgm().log(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop( autoencoder, dataloader )

        embedding: xa.DataArray = autoencoder.embedding( input_data )
        reproduction: xa.DataArray = autoencoder.reproduction(input_data)

        return ( embedding, reproduction, input_data )

    def train_loop( self, autoencoder: Autoencoder, dataloader: xaTorchDataLoader ):
        size = len( dataloader )
        for batch, (X, y) in enumerate(dataloader):
            lgm().log( f"  --> Batch: {batch}, X shape = {X.numpy().shape}" )
            self.optimizer.zero_grad()
            pred = autoencoder(X)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            lgm().log(f"  --- --> Optimize: loss = {loss}")

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def rm():
    return ReductionManager.instance()

