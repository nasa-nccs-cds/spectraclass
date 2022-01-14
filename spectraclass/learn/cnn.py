import math

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from .base import LearningModel
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


class CNNLearningModel(LearningModel):

    def __init__(self, **kwargs ):
        LearningModel.__init__(self, "svc",  **kwargs )
        self._score: Optional[np.ndarray] = None
        self._gcn = None
        self.spectral_graphs = {}
        self.spatial_graphs = {}

    def fit( self, X: np.ndarray, y: np.ndarray, **kwargs ):       # X[n_samples, n_features], y[n_samples]
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.data.base import DataManager, dm
        nHidden = kwargs.get('nhidden', 32)
        if self._gcn is None:
            nClasses = y.max() - 1
            self._gcn = GCN( X.shape[1], nHidden, nClasses )
        t0 = time.time()
        model_data: xa.DataArray = dm().getModelData()
        class_data: xa.DataArray = lm().getLabelsArray()
        train_mask: torch.tensor = torch.from_numpy( class_data.values > 0 )
        test_mask:  torch.tensor = torch.from_numpy( class_data.values == 0 )
        nodata_mask:  torch.tensor = torch.from_numpy( class_data.values == class_data.attrs['_FillValue'] )
        class_masks: Dict[str, torch.tensor] = dict( train_mask=train_mask, test_mask=test_mask, nodata_mask=nodata_mask)

        graph_data = self.getGraphData( model_data.values, class_data.values, class_masks )
        self._gcn.train_model( graph_data, **kwargs )
        print(f"Completed GCN fit, in {time.time()-t0} secs")


    def predict( self, X: np.ndarray, **kwargs ) -> np.ndarray:
        graph_data = self.getGraphData( X )
        (pred, acc) = self._gcn.evaluate_model( graph_data )
        return pred.numpy()

    def getMasks(self, class_data: np.ndarray, num_class_exemplars: int) -> Dict[str, torch.tensor]:
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

    def getGraphData( self, model_data: np.ndarray, class_data: np.ndarray = None, class_masks: Dict[str, np.ndarray] = None, **kwargs ) -> Data:
        from spectraclass.data.base import DataManager, dm
        sd = dm().getSpatialDims()
        nnspatial = kwargs.get( 'nnspatial', 8 )
        nnspectral = kwargs.get('nnspectral', 0 )
        cosine = kwargs.get( 'cosine', False )                           # only available on GPU
        node_data: torch.tensor = torch.from_numpy( model_data )
        edge_index: torch.tensor = None
        pos: torch.tensor = None
        edge_weights = None

        if nnspectral > 0:
            edge_index = self.spectral_graphs.setdefault( id(model_data), knn_graph( node_data, nnspectral, cosine=cosine ) )

        if nnspatial > 0:
            I: np.array = np.array(range(sd['ny'] * sd['nx']))
            X, Y = I % sd['nx'], I // sd['nx']
            pos: torch.tensor = torch.from_numpy(np.vstack([Y, X]).transpose())
            spatial_edge_index = self.spatial_graphs.setdefault( id(model_data), knn_graph( pos, nnspatial ) )
            edge_index: torch.tensor = spatial_edge_index if edge_index is None else torch.cat([ edge_index, spatial_edge_index], dim=1 )

        class_tensor: torch.tensor = torch.from_numpy(class_data) - 1
        graph_data = Data(x=node_data, y=class_tensor, pos=pos, edge_index=edge_index, edge_weights=edge_weights )
        nfeatures = graph_data.num_node_features

        print(f"reduced_spectral_data shape = {model_data.shape}")
        print(f"num_nodes = {graph_data.num_nodes}")
        print(f"num_edges = {graph_data.num_edges}")
        print(f"num_node_features = {nfeatures}")
        print(f"num_edge_features = {graph_data.num_edge_features}")
        print(f"contains_isolated_nodes = {graph_data.contains_isolated_nodes()}")
        print(f"contains_self_loops = {graph_data.contains_self_loops()}")
        print(f"is_directed = {graph_data.is_directed()}")

        if class_masks is not None:
            for mid, mask in class_masks.items(): graph_data[mid] = mask
        graph_data['nclasses'] = class_data.max()
        graph_data['nfeatures'] = nfeatures
        return graph_data

