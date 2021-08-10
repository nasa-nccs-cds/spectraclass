from typing import List, Union, Tuple, Optional, Dict
from torch.utils.data import DataLoader
import torch.nn.functional as F
import xarray as xa
import torch
import torch.nn as nn
import numpy as np, time, traceback
from spectraclass.data.loaders import xaTorchDataset
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
        npX: np.ndarray = x.numpy()
        lgm().log(f"  --> forward: x shape = {npX.shape}, dtype = {npX.dtype}")
        for lid in self._ops:
            layer = getattr( self, lid )
            x = layer(x)
        return x

    @classmethod
    def getLoader( cls, input: xa.DataArray, batch_size, shuffle ) -> Optional[DataLoader]:
        if input is None: return None
        dataset = xaTorchDataset( input, input )
        return DataLoader( dataset, batch_size, shuffle )

class ReductionManager(SCSingletonConfigurable):
    learning_rate = tl.Float(0.001).tag(config=True,sync=True)
    reduction_factor = tl.Float(2).tag(config=True,sync=True)
    batch_size = tl.Int(64).tag(config=True,sync=True)

    def __init__(self, **kwargs):
        super(ReductionManager, self).__init__(**kwargs)
        self._dsid = None
        self.conf = kwargs
        self._samples_coord = None

    @exception_handled
    def reduce(self, input_data: xa.DataArray, ndim: int, nepochs: int = 2 ) -> Tuple[xa.DataArray, xa.DataArray, xa.DataArray]:
        with xa.set_options(keep_attrs=True):
            return self.autoencoder_reduction( input_data, ndim, nepochs )

    def autoencoder_reduction(self, input_data: xa.DataArray, ndim: int, epochs: int, **kwargs ) -> Tuple[xa.DataArray, xa.DataArray, xa.DataArray]:
        input_dims = input_data.shape[1]
        ispecs: List[np.ndarray] = [input_data.data.max(0), input_data.data.min(0), input_data.data.mean(0), input_data.data.std(0)]
        print(f" autoencoder_reduction: input_data shape = {input_data.shape}, learning_rate = {self.learning_rate} ")
        print(f"   ----> max = {ispecs[0][:64].tolist()} " )
        print(f"   ----> min = {ispecs[1][:64].tolist()} " )
        print(f"   ----> ave = {ispecs[2][:64].tolist()} " )
        print(f"   ----> std = {ispecs[3][:64].tolist()} " )

        shuffle: bool = bool( kwargs.get('shuffle', True ) )
        autoencoder = Autoencoder( )
        autoencoder.build( input_dims, ndim, self.reduction_factor )
        dataloader: DataLoader = Autoencoder.getLoader( input_data, self.batch_size, shuffle )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam( autoencoder.parameters(), lr=self.learning_rate )

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop( autoencoder, dataloader )

        embedding: xa.DataArray = autoencoder.embedding( input_data )
        reproduction: xa.DataArray = autoencoder.reproduction(input_data)

        return ( embedding, reproduction, input_data )

    def train_loop( self, autoencoder: Autoencoder, dataloader: DataLoader ):
        size = len( dataloader.dataset )
        for batch, (X, y) in enumerate(dataloader):
            self.optimizer.zero_grad()
            pred = autoencoder(X)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            if batch % 10 == 0:
                npX: np.ndarray = X.numpy()
                print(f"  --> Batch: {batch}, X shape = {npX.shape}, dtype = {npX.dtype}")
                loss, current = loss.item(), batch * len(X)
                print(f"  ** loss: {loss:>7f}  [{current:>5d}/{size}]")


def rm():
    return ReductionManager.instance()

