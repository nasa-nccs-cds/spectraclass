import time, random, numpy as np, torch
from typing import List, Union, Tuple, Optional, Dict, Callable
from pynndescent import NNDescent
from spectraclass.data.base import DataManager
import torch_geometric
import hvplot.xarray
import holoviews as hv
from spectraclass.learn.gcn import GCN
import panel as pn
from torch_geometric.data import Data
import xarray as xa

def plot_results( class_map: xa.DataArray, pred_class_map: xa.DataArray ):
    class_plot = class_map.hvplot.image(cmap='Category20')
    pred_class_plot = pred_class_map.hvplot.image( cmap='Category20' )
    pn.Row( class_plot, pred_class_plot  ).show("Indian Pines")

def calc_edge_weights( distance: np.ndarray) -> torch.tensor:
    sig = distance.std()
    x = (distance * distance) / (-2 * sig * sig)
    return torch.from_numpy(np.exp(x))

def getMasks( class_data: np.ndarray, num_class_exemplars: int ) -> Dict[str,torch.tensor]:
    nclasses: int = class_data.max()
    class_masks: List[np.ndarray] = [(class_data == (iC + 1)) for iC in range(nclasses)]
    test_mask: np.ndarray = (class_data > 0)
    nodata_mask = np.logical_not(test_mask)
    class_indices = [np.argwhere(class_masks[iC]).flatten() for iC in range(nclasses)]
    train_class_indices = [np.random.choice(class_indices[iC], size=num_class_exemplars, replace=False) for iC in range(nclasses)]
    train_indices = np.hstack(train_class_indices)
    train_mask = np.full( test_mask.shape, False, dtype=bool )
    train_mask[train_indices] = True
    test_mask[train_indices] = False
    return dict( train_mask = torch.from_numpy(train_mask),
                 test_mask = torch.from_numpy(test_mask),
                 nodata_mask = torch.from_numpy(nodata_mask) )

def getGraphData( trial: int, project_data: xa.Dataset, class_data: np.ndarray, class_masks: Dict[str,np.ndarray], **kwargs ) -> Data:
    nnspatial = kwargs.get( 'nnspatial', 8 )
    nnspectral = kwargs.get('nnspectral', 0 )
    cosine = kwargs.get( 'cosine', False )                           # only available on GPU
    reduced_spectral_data: xa.DataArray = project_data['reduction']
    raw_spectral_data: xa.DataArray = project_data['raw']
    node_data: torch.tensor = torch.from_numpy(reduced_spectral_data.values)
    edge_index: torch.tensor = None
    pos: torch.tensor = None
    edge_weights = None

    if nnspectral > 0:
        edge_index = torch_geometric.nn.knn_graph( node_data, nnspectral, cosine=cosine )

    if nnspatial > 0:
        [ny, nx] = raw_spectral_data.shape[1:]
        I: np.array = np.array(range(ny * nx))
        X, Y = I % nx, I // nx
        pos: torch.tensor = torch.from_numpy(np.vstack([Y, X]).transpose())
        spatial_edge_index = torch_geometric.nn.knn_graph( pos, nnspatial )
        edge_index: torch.tensor = spatial_edge_index if edge_index is None else torch.cat([ edge_index, spatial_edge_index], dim=1 )

    class_tensor: torch.tensor = torch.from_numpy(class_data) - 1
    graph_data = Data(x=node_data, y=class_tensor, pos=pos, edge_index=edge_index, edge_weights=edge_weights )
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

    for mid, mask in class_masks.items(): graph_data[mid] = mask
    graph_data['nclasses'] = class_data.max()
    graph_data['nfeatures'] = nfeatures
    return graph_data

nhidden = 32
sgd_parms = dict( nepochs = 500, lr = 0.02, weight_decay = 0.0005, dropout = True )
G_parms = dict( nnspatial = 8, nnspectral = 0, cosine = False )     # 'cosine' only available on GPU
ntrials = 1
num_class_exemplars = 5
accuracy = []

dm: DataManager = DataManager.initialize( "salinas", 'aviris' )
project_data: xa.Dataset = dm.loadCurrentProject( "main" )
class_map: xa.DataArray = dm.getClassMap()
class_data: np.ndarray = class_map.values.flatten().astype(np.compat.long)
best_model = None
best_acc = 0.0
best_graph_data = None
ts = time.time()

for iT in range(ntrials):
    t0 = time.time()
    class_masks: Dict[str, torch.tensor] = getMasks(class_data, num_class_exemplars)
    graph_data = getGraphData( iT, project_data, class_data, class_masks, **G_parms )
    model = GCN( graph_data.nfeatures, nhidden, graph_data.nclasses )
    GCN.train_model( model, graph_data, **sgd_parms )
    ( pred, acc ) = GCN.evaluate_model( model, graph_data )
    accuracy.append( acc )
    print(f" ** Completed trial {iT}/{ntrials} in {time.time()-t0} sec: Accuracy = {acc}, running average = {np.array(accuracy).mean()}")
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_graph_data = graph_data

acc_data = np.array(accuracy)
( pred, acc ) = GCN.evaluate_model( best_model, best_graph_data )
print( f"Average accuracy over {ntrials} trials = {acc_data.mean()}, std = {acc_data.std()}, total training time = {(time.time()-ts)/60} min")
pred_class_map: xa.DataArray = class_map.copy( data = pred.reshape( class_map.shape ) )
plot_results( class_map, pred_class_map )





