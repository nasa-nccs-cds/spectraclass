import random, numpy as np, torch
from typing import List, Union, Tuple, Optional, Dict, Callable
from pynndescent import NNDescent
from spectraclass.data.base import DataManager
from torch_geometric.transforms import KNNGraph
import hvplot.xarray
import holoviews as hv
from spectraclass.learn.gcn import GCN
from spectraclass.learn.mlp import MLP
import panel as pn
from torch_geometric.data import Data
import xarray as xa

dm: DataManager = DataManager.initialize( "indianPines", 'aviris' )
project_data: xa.Dataset = dm.loadCurrentProject( "main" )

def getGraphData( trial: int ) -> Data:
    knn_transform = KNNGraph(4)
    reduced_spectral_data: xa.DataArray = project_data['reduction']
    raw_spectral_data: xa.DataArray = project_data['raw']
    class_map: xa.DataArray = dm.getClassMap()
    node_data: torch.tensor = torch.from_numpy(reduced_spectral_data.values)
    spectral_grid: Data = knn_transform(Data(pos=node_data))
    spectral_graph_indices = spectral_grid.edge_index
    edge_weights = None

    [ny, nx] = raw_spectral_data.shape[1:]
    I: np.array = np.array(range(ny * nx))
    X, Y = I % nx, I // nx
    pos: torch.tensor = torch.from_numpy(np.vstack([Y, X]).transpose())
    spatial_grid: Data = knn_transform(Data(pos=pos))

    class_data: torch.tensor = torch.from_numpy(class_map.values.flatten().astype(np.compat.long)) - 1
    edge_index: torch.tensor = torch.cat([spectral_graph_indices, spatial_grid.edge_index], dim=1)
    graph_data = Data(x=node_data, y=class_data, pos=pos, edge_index=edge_index, edge_weights=edge_weights)
    nfeatures = graph_data.num_node_features

    if trial == 0:
        print(f"raw_spectral_data shape = {raw_spectral_data.shape}")
        print(f"reduced_spectral_data shape = {reduced_spectral_data.shape}")
        print(f"num_nodes = {graph_data.num_nodes}")
        print(f"num_edges = {graph_data.num_edges}")
        print(f"num_node_features = {nfeatures}")
        print(f"num_edge_features = {graph_data.num_edge_features}")
        print(f"contains_isolated_nodes = {graph_data.contains_isolated_nodes()}")
        print(f"contains_self_loops = {graph_data.contains_self_loops()}")
        print(f"is_directed = {graph_data.is_directed()}")

    num_class_exemplars = 5
    class_data: np.ndarray = class_map.values.flatten()
    nclasses: int = class_map.values.max()
    class_masks: List[np.ndarray] = [(class_data == (iC + 1)) for iC in range(nclasses)]
    test_mask: np.ndarray = (class_data > 0)
    nodata_mask = np.logical_not(test_mask)
    class_indices = [np.argwhere(class_masks[iC]).flatten() for iC in range(nclasses)]
    train_class_indices = [np.random.choice(class_indices[iC], size=num_class_exemplars, replace=False) for iC in range(nclasses)]
    train_indices = np.hstack(train_class_indices)
    train_mask = np.full([node_data.shape[0]], False, dtype=bool)
    train_mask[train_indices] = True
    test_mask[train_indices] = False

    graph_data['train_mask'] = torch.from_numpy(train_mask)
    graph_data['test_mask'] = torch.from_numpy(test_mask)
    graph_data['nodata_mask'] = torch.from_numpy(nodata_mask)
    graph_data['nclasses'] = nclasses
    graph_data['nfeatures'] = nfeatures
    return graph_data

nhidden = 32
sgd_parms = dict( nepochs = 400, lr = 0.01, weight_decay = 0.0005, dropout = True )
MODEL = GCN
ntrials = 100
use_edge_weights = False
accuracy = []

for iT in range(ntrials):
    graph_data = getGraphData( iT )
    model = MODEL( graph_data.nfeatures, nhidden, graph_data.nclasses )
    MODEL.train_model( model, graph_data, **sgd_parms )
    ( pred, acc ) = MODEL.evaluate_model( model, graph_data )
    print(f" ** Completed trial {iT}/{ntrials}: Accuracy = {acc}")
    accuracy.append( acc )

acc_data = np.array(accuracy)
print( f"Average accuracy over {ntrials} trials = {acc_data.mean()}, std = {acc_data.std()}")




