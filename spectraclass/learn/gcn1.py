import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCN1(torch.nn.Module):
    def __init__( self, num_features: int, num_hiddex: int, num_classes: int ):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv( num_features, num_hiddex )
        self.conv2 = GCNConv( num_hiddex, num_classes )

    def forward( self, data: Data ):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    @staticmethod
    def train_model( model: "GCN1", data: Data, **kwargs ):
        lr = kwargs.get('lr',0.01)
        weight_decay = kwargs.get('weight_decay', 5e-4)
        nepochs = kwargs.get( 'nepochs', 1000 )
        optimizer = torch.optim.Adam(model.parameters())  # , lr=lr, weight_decay=weight_decay )
        model.train()
        for epoch in range(nepochs):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 25 == 0:
                print(f'epoch: {epoch}, loss = {loss.data}' )

    @staticmethod
    def evaluate_model( model: "GCN1", data: Data ):
        model.eval()
        _, pred = model(data).max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
        print('Accuracy: {:.4f}'.format(acc))






