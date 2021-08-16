import random, numpy as np, torch
from typing import List, Union, Tuple, Optional, Dict, Callable
from pynndescent import NNDescent
from spectraclass.data.base import DataManager
from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
import hvplot.xarray
import holoviews as hv
from spectraclass.learn.gcn import GCN
from spectraclass.learn.mlp import MLP
import panel as pn
from torch_geometric.data import Data
import xarray as xa

dm: DataManager = DataManager.initialize( "indianPines", 'aviris' )
project_data: xa.Dataset = dm.loadCurrentProject( "main" )

def getGraphData( use_edge_weights ) -> Data:
    flow: ActivationFlow = afm().getActivationFlow( True )
    graph: NNDescent = flow.getGraph()
    D: np.ndarray = graph.neighbor_graph[1].flatten()
    reduced_spectral_data: xa.DataArray = project_data['reduction']
    raw_spectral_data: xa.DataArray = project_data['raw']
    class_map: xa.DataArray = dm.getClassMap()
    edge_weights = GCN.calc_edge_weights( D ) if use_edge_weights else None

#    edge_attr: torch.tensor  = torch.from_numpy( D.reshape( D.size, 1 ) )
    edge_index: torch.tensor = flow.getEdgeIndex()
    node_data: torch.tensor = torch.from_numpy( reduced_spectral_data.values )
    class_data: torch.tensor = torch.from_numpy( class_map.values.flatten().astype( np.compat.long ) ) - 1
    spatial_grid = None
    graph_data = Data( x=node_data, y=class_data, edge_index=edge_index, edge_weights=edge_weights, pos = spatial_grid )
    nfeatures = graph_data.num_node_features

    print( f"raw_spectral_data shape = {raw_spectral_data.shape}")
    print( f"reduced_spectral_data shape = {reduced_spectral_data.shape}")
    print( f"num_nodes = {graph_data.num_nodes}")
    print( f"num_edges = {graph_data.num_edges}")
    print( f"num_node_features = {nfeatures}")
    print( f"num_edge_features = {graph_data.num_edge_features}")
    print( f"contains_isolated_nodes = {graph_data.contains_isolated_nodes()}")
    print( f"contains_self_loops = {graph_data.contains_self_loops()}")
    print( f"is_directed = {graph_data.is_directed()}")

    num_class_exemplars = 5
    class_data: np.ndarray = class_map.values.flatten()
    nclasses: int = class_map.values.max()
    class_masks: List[np.ndarray] = [ (class_data == (iC+1) ) for iC in range(nclasses) ]
    test_mask: np.ndarray = (class_data > 0)
    nodata_mask = np.logical_not( test_mask )
    class_indices = [ np.argwhere(class_masks[iC]).flatten() for iC in range(nclasses) ]
    train_class_indices = [  np.random.choice(class_indices[iC], size=num_class_exemplars, replace=False )  for iC in range(nclasses)  ]
    train_indices = np.hstack( train_class_indices )
    train_mask = np.full( [ node_data.shape[0] ], False, dtype=bool )
    train_mask[ train_indices ] = True
    test_mask[ train_indices ] = False

    graph_data['train_mask'] = torch.from_numpy( train_mask )
    graph_data['test_mask'] = torch.from_numpy( test_mask )
    graph_data['nodata_mask'] = torch.from_numpy( nodata_mask )
    graph_data['nfeatures'] = nfeatures
    graph_data['nclasses'] =nclasses
    return graph_data

nhidden = 32
sgd_parms = dict( nepochs = 1000, lr = 0.02, weight_decay = 0.0005, dropout = True )
MODEL = GCN
ntrials = 5
use_edge_weights = False
accuracy = []

for iT in range(ntrials):
    graph_data = getGraphData( use_edge_weights )
    model = MODEL( graph_data.nfeatures, nhidden, graph_data.nclasses )
    MODEL.train_model( model, graph_data, **sgd_parms )
    ( pred, acc ) = MODEL.evaluate_model( model, graph_data )
    accuracy.append( acc )

acc_data = np.array(accuracy)
print( f"Average accuracy over {ntrials} trials = {acc_data.mean()}, std = {acc_data.std()}")




