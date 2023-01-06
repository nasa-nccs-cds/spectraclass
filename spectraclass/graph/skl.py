from sklearn.neighbors import NearestNeighbors as skNearestNeighbors
# from pynndescent import NNDescent
import numpy as np
from .manager import ActivationFlow, afm
import xarray as xa
import numba as nb
from typing import List, Union, Tuple, Optional, Dict
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm
import os, time, traceback

@nb.njit( fastmath=True,
    locals={
        "selection": nb.types.Array(nb.types.boolean, 1, 'C'),
        "gindices": nb.types.Array(nb.types.int32, 1, 'C'),
        "labels": nb.types.Array(nb.types.int32, 1, 'C'),
        "index_stack": nb.types.Array(nb.types.int32, 2, 'F'),
    },)
def getFilteredLabels( labels: np.ndarray ) -> np.ndarray:
    indices = np.arange(labels.shape[0], dtype = np.int32 )
    selection = (labels > 0)
    index_stack = np.vstack( (indices, labels) ).transpose()
    return np.copy( index_stack[ selection ] )

@nb.njit( fastmath=True,
    locals={
        "iN": nb.types.int32,
        "polyId": nb.types.int32,
        "pid1": nb.types.int64,
        "I": nb.types.Array(nb.types.int64, 2, 'C'),
        "label_spec": nb.types.Array(nb.types.int32, 1, 'A'),
        "C": nb.types.Array(nb.types.int32, 1, 'C'),
        "CS": nb.types.Array(nb.types.int32, 1, 'C'),
        "FC": nb.types.Array(nb.types.int32, 2, 'C'),
        "P": nb.types.Array(nb.types.float32, 1, 'C'),
        "D": nb.types.Array(nb.types.float32, 2, 'C'),
    },)
def iterate_spread_labels( I: np.ndarray, D: np.ndarray, C: np.ndarray, P: np.ndarray, bidirectional: bool ):
    if bidirectional:
        for iN in np.arange( 1, I.shape[1], dtype=np.int32 ):
            CS = np.copy( C[I[:,iN]] )
            FC = getFilteredLabels( CS )
            for label_spec in FC:
                pid = label_spec[0]
                pid1 = I[pid, iN]
                PN = P[pid1] + D[pid1, iN]
                if (C[pid] == 0) or (PN < P[pid]):
                    C[pid] = label_spec[1]
                    P[pid] = PN
    FC = getFilteredLabels( C )
    for iN in np.arange( 1, I.shape[1], dtype=np.int32 ):
        for label_spec in FC:
            pid = label_spec[0]
            pid1 = I[pid, iN]
            PN = P[pid] + D[pid, iN]
            if (C[pid1] == 0) or (PN < P[pid1]):
                C[pid1] = label_spec[1]
                P[pid1] = PN

class skActivationFlow(ActivationFlow):

    def __init__(self, nodes_data: xa.DataArray, n_neighbors: int, **kwargs ):
        ActivationFlow.__init__( self,  n_neighbors, **kwargs )
        self._knn_graph: skNearestNeighbors = None
        self.I: np.ndarray = None
        self.D: np.ndarray = None
        self.P: np.ndarray = None
        self.C: np.ndarray = None
        self.setNodeData( nodes_data )

    def query(self, X: np.ndarray, n_neighbors: int, **kwargs ) -> Tuple[np.ndarray,np.ndarray]:
        return self._knn_graph.kneighbors( X, n_neighbors, return_distance=True )

    def get_distances(self) -> np.ndarray:
        return self.P

    def get_classes(self) -> np.ndarray:
        return self.C

    @property
    def neighbor_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.I, self.D)

    def setNodeData(self, nodes_data: xa.DataArray ):
        from spectraclass.data.spatial.tile.manager import tm
        if (nodes_data.size > 0):
            t0 = time.time()
            self.nodes = nodes_data
            self.getGraph()
            D_sk, I_sk = self._knn_graph.kneighbors( self.nodes, self.nneighbors )
            self._knn_graph.neighbor_graph = ( I_sk, D_sk )
            self.I: np.ndarray = I_sk
            self.D: np.ndarray = D_sk.astype(np.float32)
            lgm().log(f" --->  $$$D: setNodeData D=> {self.D.__class__}:{self.D.dtype}",print=True)
            dt = (time.time()-t0)
            lgm().log(f"Computed NN skGraph with {self._knn_graph.n_neighbors} neighbors and {nodes_data.shape[0]} verts in {dt} sec ({dt / 60} min)",print=True)
            lgm().trace(f"\nNN-skGraph[{tm().image_index}:{tm().block_index}]")
        else:
            lgm().log("No data available for this block")

    def getGraph(self, njobs=4 ):
        if self._knn_graph is None:
            t0 = time.time()
            self._knn_graph = skNearestNeighbors(n_jobs=njobs)
            self._knn_graph.fit( self.nodes )
            lgm().log(f"skNearestNeighbors fit in {time.time()-t0} sec, njobs = {njobs}")
        return self._knn_graph

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        sample_mask = sample_data == 0
        self.C = np.array( sample_data, dtype=np.dtype(np.int32) )
        label_count = np.count_nonzero(self.C)
        if label_count == 0:
            ufm().show( "Workflow violation: Must label some points before this algorithm can be applied", "red" )
            lgm().log(" ----> No Labeled points in spread()")
            return None
        P_init = np.full( self.C.shape, float('inf'), dtype=np.float32 )
        self.P = np.where( sample_mask, P_init, 0.0 )
        lgm().log(f"Beginning graph flow iterations, #C = {label_count}, C[:10] = {self.C[:10]}")
        t0 = time.time()
        converged = False
        for iter in range(nIter):
            try:
                iterate_spread_labels( self.I, self.D, self.C, self.P, kwargs.get('bidirectional',False) )
                new_label_count = np.count_nonzero(self.C)
                if new_label_count == label_count:
                    lgm().log("Converged!")
                    converged = True
                    break
                else:
                    label_count = new_label_count
                    lgm().log(f" -->> Iter{iter + 1}: #C = {label_count}")
            except Exception as err:
                lgm().exception(f"Error in graph flow iteration {iter}:")
                break

        t1 = time.time()
        lgm().log(f"Completed graph flow {nIter} iterations in {(t1 - t0)} sec, #marked = {np.count_nonzero(self.C)}")
        return converged


