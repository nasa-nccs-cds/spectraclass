import random, numpy as np, torch
from pynndescent import NNDescent
from spectraclass.data.base import DataManager
from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
import xarray as xa

dm: DataManager = DataManager.initialize( "demo2", 'desis' )
project_data: xa.Dataset = dm.loadCurrentProject( "main" )

flow: ActivationFlow = afm().getActivationFlow()
graph: NNDescent = flow.getGraph()
D: np.ndarray  = graph.neighbor_graph[1]     # shape [nsamples,n_neighbors]
edge_index: torch.tensor = flow.getEdgeIndex()
print( edge_index )



