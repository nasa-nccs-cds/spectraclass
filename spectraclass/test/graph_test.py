import random, numpy as np, torch
from typing import List, Union, Tuple, Optional, Dict, Callable
from pynndescent import NNDescent
from spectraclass.data.base import DataManager
from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
import hvplot.xarray
import holoviews as hv
import panel as pn
from torch_geometric.data import Data
import xarray as xa

view_band = 10

dm: DataManager = DataManager.initialize( "indianPines", 'aviris' )
project_data: xa.Dataset = dm.loadCurrentProject( "main" )
flow: ActivationFlow = afm().getActivationFlow()
graph: NNDescent = flow.getGraph()
D: np.ndarray = graph.neighbor_graph[1].flatten()
reduced_spectral_data: xa.DataArray = project_data['reduction']
raw_spectral_data: xa.DataArray = project_data['raw']
class_map: xa.DataArray = dm.getClassMap()
band_image_data: np.ndarray = reduced_spectral_data.values[:,view_band].reshape( [1] + list(raw_spectral_data.shape[1:]) )
band_image: xa.DataArray = class_map.copy( True, band_image_data )

edge_attr: torch.tensor  = torch.from_numpy( D.reshape( D.size, 1 ) )
edge_index: torch.tensor = flow.getEdgeIndex()
node_data: torch.tensor = torch.from_numpy( reduced_spectral_data.values )
graph_data = Data( x=node_data, edge_index=edge_index, edge_attr=edge_attr )

print( f"raw_spectral_data shape = {raw_spectral_data.shape}")
print( f"reduced_spectral_data shape = {reduced_spectral_data.shape}")
print( f"num_nodes = {graph_data.num_nodes}")
print( f"num_edges = {graph_data.num_edges}")
print( f"num_node_features = {graph_data.num_node_features}")
print( f"num_edge_features = {graph_data.num_edge_features}")
print( f"contains_isolated_nodes = {graph_data.contains_isolated_nodes()}")
print( f"contains_self_loops = {graph_data.contains_self_loops()}")
print( f"is_directed = {graph_data.is_directed()}")

class_plot = class_map.hvplot.image( cmap='Category20', clim=(0, 20) )

#band_plot = band_image.hvplot.image( cmap='jet', clim=( -1.5, 1.5 ) )
#pn.Row( class_plot, band_plot  ).show("Indian Pines")

nclasses: int = class_map.values.max()
class_masks: List[np.ndarray] = [ (class_map.values == (iC+1) ) for iC in range(nclasses) ]
class_indices = [ class_map.where(class_masks[iC]) for iC in range(nclasses) ]







