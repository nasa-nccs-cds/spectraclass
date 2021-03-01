from spectraclass.data.base import DataManager
import time, numpy as np
from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
import pygsp as gsp
from scipy.sparse import coo_matrix
from spectraclass.data.spatial.tile.manager import TileManager, tm
from pygsp import graphs
from graph_coarsening.coarsening_utils import *
import graph_coarsening.graph_utils
gsp.plotting.BACKEND = 'matplotlib'

method = 'variation_neighborhood'
r = 0.6  # the extend of dimensionality reduction (r=0 means no reduction)
k = 5
kmax = int(3 * k)

t0 = time.time()
dm: DataManager = DataManager.initialize("demo4",'keelin')
dm.loadCurrentProject("main")
block = tm().getBlock()
spectra, spatial_coords = block.getPointData()
nsamples = spectra.shape[0]
flow: ActivationFlow = afm().getActivationFlow()

graph = flow.getGraph()
I: np.ndarray  = graph.neighbor_graph[0]     # shape [nsamples,n_neighbors]
D: np.ndarray  = graph.neighbor_graph[1]     # shape [nsamples,n_neighbors]
nneighbors = I.shape[1]

rIndex: np.ndarray = np.broadcast_to( np.arange( 0, nsamples ).reshape( (nsamples,1) ), (nsamples,nneighbors) )
W = coo_matrix( ( D.flatten(), ( rIndex.flatten(), I.flatten() )  ), shape=(nsamples, nsamples) )

print( f"Computed nn-graph[{nneighbors}] in time {time.time()-t0}")
t1 = time.time()

G = graphs.Graph(W, coords = spectra)
C, Gc, Call, Gall = coarsen(G, K=k, r=r, method=method)
n = Gc.N

print( f"Computed Graph graph coarsening in time {time.time()-t1}")
