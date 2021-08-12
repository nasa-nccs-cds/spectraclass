import random, numpy as np
from pynndescent import NNDescent
from spectraclass.data.base import DataManager
from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
import xarray as xa

dm: DataManager = DataManager.initialize( "demo2", 'desis' )
project_data: xa.Dataset = dm.loadCurrentProject( "main" )
dm.prepare_inputs()

flow: ActivationFlow = afm().getActivationFlow()
graph: NNDescent = flow.getGraph()

I: np.ndarray  = graph.neighbor_graph[0]     # shape [nsamples,n_neighbors]
D: np.ndarray  = graph.neighbor_graph[1]     # shape [nsamples,n_neighbors]

print( I[:5] )
print( D[:5] )


