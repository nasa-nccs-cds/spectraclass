import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Tuple, Optional, Dict
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCN(torch.nn.Module):
    def __init__( self, num_features: int, num_hidden: int, num_classes: int ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv( num_features, num_hidden )
        self.conv2 = GCNConv( num_hidden, num_classes )
#        torch.nn.init.xavier_uniform(self.conv1.weight)
#        torch.nn.init.xavier_uniform(self.conv2.weight)
        self._dropout = True
        print( f"Init GCN: Base Layer weights = {self.conv1.weight.data.numpy()}")

    def set_dropout(self, active: bool ):
        self._dropout = active

    def forward( self, data: Data ):
        x, edge_index = data.x, data.edge_index
        edge_weights = data.__dict__.get( 'edge_weights', None )
        x = self.conv1( x, edge_index, edge_weights )
        x = F.relu(x)
        if self._dropout:
            x = F.dropout( x, training=self.training )
        x = self.conv2( x, edge_index, edge_weights )
        return F.log_softmax(x, dim=1)

    @classmethod
    def train_model( cls, model: "GCN", data: Data, **kwargs ):
        lr = kwargs.get('lr',0.01)
        weight_decay = kwargs.get('weight_decay', 5e-4)
        nepochs = kwargs.get( 'nepochs', 200 )
        dropout = kwargs.get( 'dropout', True )
        model.set_dropout( dropout )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay )
        print( f"Training model with lr={lr}, weight_decay={weight_decay}, nepochs={nepochs}, dropout={model._dropout}")
        model.train()
        for epoch in range(nepochs):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 25 == 0:
                print(f'epoch: {epoch}, loss = {loss.data}' )

    @classmethod
    def evaluate_model( cls, model: "GCN", data: Data ) -> Tuple[np.ndarray,float]:
        model.eval()
        _, pred = model(data).max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
        print(' --> Accuracy: {:.4f}'.format(acc))
        pred_data = pred.numpy() + 1
        pred_data[ data.nodata_mask.numpy() ] = 0
        return ( pred_data, acc )


    @classmethod
    def calc_edge_weights(cls, distance: np.ndarray ) -> torch.tensor:
        sig = distance.std()
        x = ( distance * distance ) / ( -2 * sig * sig )
        return torch.from_numpy( np.exp( x ) )






