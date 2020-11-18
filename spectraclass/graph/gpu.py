import numpy as np
import numpy.ma as ma
from .base import ActivationFlow
import xarray as xa
import numba
import cudf, cuml, cupy, cupyx
from cuml.neighbors import NearestNeighbors
from cupyx.scipy.sparse import csr_matrix
from cugraph.traversal.sssp import shortest_path
from typing import List, Union, Tuple, Optional, Dict
import os, time, threading, traceback

class gpActivationFlow(ActivationFlow):

    def __init__(self, nodes_data: xa.DataArray, n_neighbors: int, **kwargs ):
        ActivationFlow.__init__( self,  n_neighbors, **kwargs )
        self.I: cudf.DataFrame = None
        self.D: cudf.DataFrame = None
        self.P: cudf.DataFrame = None
        self.C: cudf.DataFrame = None
        self.nodes: cudf.DataFrame = None
        self.setNodeData( nodes_data, **kwargs )

    def setNodeData(self, nodes_data: xa.DataArray, **kwargs ):
        print( f"{self.__class__.__name__}[{hex(id(self))}].setNodeData: input shape = {nodes_data.shape}" )
        if self.reset or (self.nodes is None):
            if (nodes_data.size > 0):
                t0 = time.time()
                self.nodes = cudf.DataFrame({icol: nodes_data[:, icol] for icol in range(nodes_data.shape[1])})
                self.nnd = NearestNeighbors( n_neighbors=self.nneighbors )
                self.nnd.fit( self.nodes )
                self.D, self.I = self.nnd.kneighbors( self.nodes, return_distance=True)
                dt = (time.time()-t0)
                print( f"Computed NN Graph with {self.nnd.n_neighbors} neighbors and {nodes_data.shape[0]} verts in {dt} sec ({dt/60} min)")
                print( f"  ---> Indices shape = {self.I.shape}, Distances shape = {self.D.shape} "  )
            else:
                print( "No data available for this block")

    def getGraph(self):
        return None

    def getConnectionMatrix(self) -> csr_matrix:
        distances = cupy.ravel(cupy.fromDlpack( self.D.to_dlpack()) )
        indices = cupy.ravel(cupy.fromDlpack( self.I.to_dlpack()) )
        n_samples = indices.shape[0]
        n_nonzero = n_samples * self.nneighbors
        rowptr = cupy.arange( 0, n_nonzero + 1, self.nneighbors )
        knn_graph = cupyx.scipy.sparse.csr_matrix((distances, indices, rowptr), shape=(n_samples, n_samples))
        print(f"Completed KNN, sparse graph shape = {knn_graph.shape}")
        return knn_graph

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        converged = True

        spdf = shortest_path( G, source_pid )
        spdf.sort_by("vertex")
        distances = spdf["distance"]

        self.reset = False
        return converged


