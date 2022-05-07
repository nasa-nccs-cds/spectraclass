import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from typing import List, Union, Tuple, Optional, Dict

class MLP(torch.nn.Module):
    def __init__( self, num_features: int, num_hidden: int, num_classes: int ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear( num_features, num_hidden )
        self.fc2 = torch.nn.Linear( num_hidden, num_classes )

    def forward( self, data: Data ):
        x = data.x
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    @classmethod
    def train_model( cls, model: "MLP", data: Data, **kwargs ):
        lr = kwargs.get('lr',0.01)
        weight_decay = kwargs.get('weight_decay', 5e-4)
        nepochs = kwargs.get( 'nepochs', 200 )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay )
        model.train()
        for epoch in range(nepochs):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'epoch: {epoch}, loss = {loss.data}' )

    @classmethod
    def evaluate_model( cls, model: "MLP", data: Data ) -> Tuple[np.ndarray,float]:
        model.eval()
        _, pred = model(data).max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
        print(' --> Accuracy: {:.4f}'.format(acc))
        pred_data = pred.numpy() + 1
        pred_data[ data.nodata_mask.numpy() ] = 0
        return ( pred_data, acc )