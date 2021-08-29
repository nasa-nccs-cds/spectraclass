import math

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import xarray as xa
import numpy as np
from spectraclass.data.base import DataManager
from typing import List, Union, Tuple, Optional, Dict

def lname( iL: int ): return f"layer-{iL}"

class CNN(torch.nn.Module):
    def __init__( self, num_layer_features: List[int], num_classes: int, kernel_sizes: List[int], **cnn_parms ):
        super(CNN, self).__init__()
        self._LS = [ *num_layer_features ]
        self._LS.append( num_classes )
        self.nLayers =  len(self._LS) - 1
        self._ks = kernel_sizes
        self._attrs = cnn_parms
        self._dropout = True
        self.build_network()

    def set_layer(self, iL: int, layer: torch.nn.Conv2d ):
        setattr(self, lname(iL), layer )

    def get_layer(self, iL: int ) -> torch.nn.Conv2d:
        return getattr( self, lname(iL) )

    def build_network( self ):
        for iL in range( self.nLayers ):
            pd = (self._ks[iL] - 1) // 2
            layer = torch.nn.Conv2d( self._LS[iL], self._LS[iL+1], ( self._ks[iL], self._ks[iL] ), padding=(pd,pd) )
            self.set_layer( iL, layer )

    def forward( self, data: Data ):
        x: torch.Tensor = data.x
        for iL in range( self.nLayers ):
            layer = self.get_layer( iL )
            last_layer =  (iL == (self.nLayers - 1) )
            if last_layer and self._dropout and (iL > 0):
                x = F.dropout( x, training=self.training )
            x = layer(x)
            x = F.log_softmax( x, dim=1 ) if last_layer else F.relu( x )
        return self.to_linear(x)

    def to_linear(self, x: torch.Tensor) -> torch.Tensor:
        return x.resize( x.shape[1], x.shape[2]*x.shape[3] ).transpose(0,1)

    def train_model( self, data: Data, **kwargs ):
        lr = kwargs.get('lr',0.01)
        weight_decay = kwargs.get('weight_decay', 5e-4)
        nepochs = kwargs.get( 'nepochs', 200 )
        self._dropout = kwargs.get( 'dropout', True )
        optimizer = torch.optim.Adam( self.parameters(), lr=lr, weight_decay=weight_decay )
        self.train()
        for epoch in range(nepochs):
            optimizer.zero_grad()
            out = self(data)
            loss = F.nll_loss( out[data.train_mask], data.y[data.train_mask] )
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'epoch: {epoch}, loss = {loss.data}' )

    def npnorm(self, data: np.ndarray, axis: int ):
        v0, v1 = data.min(axis=axis).elem(), data.max(axis=axis).elem()
        return ( data - v0 ) / ( v1 - v0 )

    def evaluate_model( self, data: Data ) -> Tuple:
        self.eval()
        class_results = self(data)
        rvals, pred = class_results.max(dim=1)
        reliability = rvals.detach().numpy()
        correct = int( pred[data.test_mask].eq( data.y[data.test_mask] ).sum().item() )
        acc = correct / int( data.test_mask.sum() )
        pred_data = pred.numpy() + 1
        pred_data[ data.nodata_mask.numpy() ] = 0
        print( f"Class prob range = ({class_results.min()}, {class_results.max()}), mean = {class_results.mean()}, std = {class_results.std()}")
        return ( pred_data, reliability, acc )

    @staticmethod
    def getMasks( class_map: np.ndarray, num_class_exemplars: int) -> Dict[str, torch.tensor]:
        class_data = class_map.flatten()
        nclasses: int = class_data.max()
        class_masks: List[np.ndarray] = [(class_data == (iC + 1)) for iC in range(nclasses)]
        test_mask: np.ndarray = (class_data > 0)
        nodata_mask = np.logical_not(test_mask)
        class_indices = [np.argwhere(class_masks[iC]).flatten() for iC in range(nclasses)]
        train_class_indices = [np.random.choice(class_indices[iC], size=num_class_exemplars, replace=False) for iC in
                               range(nclasses)]
        train_indices = np.hstack(train_class_indices)
        train_mask = np.full(test_mask.shape, False, dtype=bool)
        train_mask[train_indices] = True
        test_mask[train_indices] = False
        return dict(train_mask=torch.from_numpy(train_mask),
                    test_mask=torch.from_numpy(test_mask),
                    nodata_mask=torch.from_numpy(nodata_mask))

    @staticmethod
    def getConvData( dm: DataManager) -> torch.Tensor:
        project_data: xa.Dataset = dm.loadCurrentProject("main")
        reduced_spectral_data: xa.DataArray = project_data['reduction']
        raw_spectral_data: xa.DataArray = project_data['raw']
        Nf = reduced_spectral_data.shape[1]
        Nw = raw_spectral_data.shape[2]
        Nh = raw_spectral_data.shape[1]
        X = reduced_spectral_data.values.transpose().reshape([1, Nf, Nh, Nw])
        return torch.from_numpy(X)