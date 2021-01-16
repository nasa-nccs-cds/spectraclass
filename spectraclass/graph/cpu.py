from pynndescent import NNDescent
import numpy as np
from .manager import ActivationFlow, afm
import xarray as xa
import numba
from typing import List, Union, Tuple, Optional, Dict
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm
import os, time, traceback

@numba.njit(fastmath=True,
    locals={
        "selection": numba.boolean[:],
        "indices": numba.int32[:],
        "labels": numba.int32[:],
        "index_stack": numba.int32[:,:],
    },)
def getFilteredLabels( labels: np.ndarray ) -> np.ndarray:
    indices = np.arange(labels.shape[0], dtype = np.int32 )
    selection = (labels > 0)
    index_stack = np.vstack( (indices, labels) ).transpose()
    return index_stack[ selection ]

@numba.jit(fastmath=True,
    locals={
        "iN": numba.int32,
        "pid": numba.int32,
        "pid1": numba.int64,
        "I": numba.int64[:,:],
        "label_spec": numba.int32[:],
        "C": numba.int32[:],
        "P": numba.float32[:],
        "D": numba.float32[:,:],
    },)
def iterate_spread_labels( I: np.ndarray, D: np.ndarray, C: np.ndarray, P: np.ndarray ):
    for iN in np.arange( 1, I.shape[1], dtype=np.int32 ):
        FC = getFilteredLabels( C[I[:,iN]] )
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

class cpActivationFlow(ActivationFlow):

    def __init__(self, nodes_data: xa.DataArray, n_neighbors: int, **kwargs ):
        ActivationFlow.__init__( self,  n_neighbors, **kwargs )
        self.nnd: NNDescent = None
        self.I: np.ndarray = None
        self.D: np.ndarray = None
        self.P: np.ndarray = None
        self.C: np.ndarray = None
        self.setNodeData( nodes_data )

    def get_distances(self) -> np.ndarray:
        return self.P

    def get_classes(self) -> np.ndarray:
        return self.C

    def setNodeData(self, nodes_data: xa.DataArray ):
        if (nodes_data.size > 0):
            t0 = time.time()
            self.nodes = nodes_data
            self.getGraph()
            self.I: np.ndarray = self._knn_graph.neighbor_graph[0]
            self.D: np.ndarray = self._knn_graph.neighbor_graph[1].astype(np.float32)
            lgm().log(f" --->  $$$D: setNodeData D=> {self.D.__class__}:{self.D.dtype}")
            dt = (time.time()-t0)
            lgm().log(f"Computed NN Graph with {self._knn_graph.n_neighbors} neighbors and {nodes_data.shape[0]} verts in {dt} sec ({dt / 60} min)")
        else:
            lgm().log("No data available for this block")

    def getGraph(self):
        if self._knn_graph is None:
            n_trees =  5 + int(round((self.nodes.shape[0]) ** 0.5 / 20.0))
            n_iters =  max(5, 2 * int(round(np.log2(self.nodes.shape[0]))))
            kwargs = dict( n_trees=n_trees, n_iters=n_iters, n_neighbors=self.nneighbors, max_candidates=60, verbose=True, metric = self.metric )
            if self.metric == "minkowski": kwargs['metric_kwds'] = dict( p=self.p )
            lgm().log(f"Computing NN-Graph with parms= {kwargs}")
            self._knn_graph = NNDescent( self.nodes.values, **kwargs )
        return self._knn_graph

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        sample_mask = sample_data == 0
        self.C = np.array( sample_data, dtype=np.dtype(np.int32) )
        label_count = np.count_nonzero(self.C)
        if label_count == 0:
            ufm().show( "Workflow violation: Must label some points before this algorithm can be applied", "red" )
            return None
        P_init = np.full( self.C.shape, float('inf'), dtype=np.float32 )
        self.P = np.where( sample_mask, P_init, 0.0 )
        lgm().log(f"Beginning graph flow iterations, #C = {label_count}, C[:10] = {self.C[:10]}")
        t0 = time.time()
        converged = False
        for iter in range(nIter):
            try:
#                for iX, X in enumerate([ self.I, self.D, self.C, self.P ]): lgm().log(f" I{iX} -> {X.shape}:{X.dtype}")
                iterate_spread_labels( self.I, self.D, self.C, self.P )
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


