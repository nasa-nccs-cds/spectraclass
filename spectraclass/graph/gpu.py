import numpy as np
import numpy.ma as ma
from .manager import ActivationFlow
import xarray as xa
import numba
from spectraclass.util.logs import LogManager, lgm
import cudf, cuml, cupy, cupyx, cugraph
from cupyx.scipy.sparse import csr_matrix
from cugraph.traversal.sssp import shortest_path
from cuml.neighbors import NearestNeighbors
from typing import List, Union, Tuple, Optional, Dict
from spectraclass.application.controller import app
import os, time, threading, traceback
USE_SKLEARN = False

class gpActivationFlow(ActivationFlow):

    def __init__(self, nodes_data: xa.DataArray, n_neighbors: int, **kwargs ):
        ActivationFlow.__init__( self,  n_neighbors, **kwargs )
        app().show_gpu_usage()
        self.I: cudf.DataFrame = None
        self.D: cudf.DataFrame = None
        self.P: cudf.DataFrame = None
        self.C: cudf.DataFrame = None
        self.nnd: cuml.neighbors.NearestNeighbors = None
        self.nodes = None
        self.n_samples = None
        self.setNodeData( nodes_data, **kwargs )

    def get_distances(self) -> np.ndarray:
        return self.P

    def get_classes(self) -> np.ndarray:
        return self.C

    def setNodeData(self, node_data: xa.DataArray, **kwargs ):
        input_data = node_data.values
        reset = kwargs.get( 'reset', False )
        lgm().log( f"{self.__class__.__name__}[{hex(id(self))}].setNodeData: input shape = {input_data.shape}" )
        if reset or (self.I is None):
            if (input_data.size > 0):
                t0 = time.time()

                # if USE_SKLEARN:
                #     from sklearn.neighbors import NearestNeighbors
                #     self.nnd = NearestNeighbors( n_neighbors=self.nneighbors, n_jobs=-1 )
                #     self.nnd.fit( input_data )
                #     D_sk, I_sk = self.nnd.kneighbors( input_data, return_distance=True )
                #     self.D, self.I = cudf.DataFrame( D_sk ), cudf.DataFrame( I_sk )
                # else:

                self.nodes = cudf.DataFrame( input_data )
                self.n_samples = input_data.shape[0]
                lgm().log( f"NearestNeighbors{input_data.shape}: input nodes = {self.nodes.head(10)}")
                self.nnd = NearestNeighbors( n_neighbors=self.nneighbors, metric=self.metric, p=self.p )
                self.nnd.fit( input_data )
                self.D, self.I = self.nnd.kneighbors( self.nodes, return_distance=True)
                dt = (time.time()-t0)
                lgm().log( f"Computed NN Graph with {self.nneighbors} neighbors and {input_data.shape[0]} verts in {dt} sec ({dt/60} min)")
                lgm().log( f"  ---> Indices shape = {self.I.shape}, Distances shape = {self.D.shape}\n Indices = {self.I.head(10)} "  )
            else:
                lgm().log( "No data available for this block")

    def getGraph(self) -> csr_matrix:
        return self.getConnectionMatrix()

    def getConnectionMatrix(self) -> csr_matrix:
        if self._knn_graph is None:
            distances = cupy.ravel(cupy.fromDlpack( self.D.to_dlpack()) )
            indices = cupy.ravel(cupy.fromDlpack( self.I.to_dlpack()) )
            n_samples = indices.shape[0]
            n_nonzero = n_samples * self.nneighbors
            rowptr = cupy.arange( 0, n_nonzero + 1, self.nneighbors )
            self._knn_graph = cupyx.scipy.sparse.csr_matrix((distances, indices, rowptr), shape=(n_samples, n_samples))
            lgm().log(f"Completed KNN, sparse graph shape = {self._knn_graph.shape}")
        return self._knn_graph

    def get_offset_series( self ):
        ishp = self.I.shape
        indices = np.repeat( np.arange( 0, ishp[0] ).reshape(ishp[0],1), ishp[1], axis = 1 )
        return cupy.ravel( indices )

    def cuIndices(self):
        indices = cupy.fromDlpack(self.I.to_dlpack())
        iseries =  cupy.ravel(indices)
        lgm().log( f" cuIndices: shape = {indices.shape}, avals = {indices[0:10]}")
        return iseries

    def spread( self, sample_data: np.ndarray, nIter: int = 1, **kwargs ) -> Optional[bool]:
        converged = True
        t0 = time.time()
        source_pid: int = sample_data[0]

        lgm().log( f" ActivationFlow:  ")
        lgm().log(f" --> I.shape = {self.I.shape}")
        lgm().log(f" --> I = {self.I.head(10)}")
        lgm().log(f" --> D = {self.D.head(10)}")

        offsets   = self.get_offset_series()
        distances = cupy.ravel( cupy.fromDlpack( self.D.to_dlpack() ) )
        indices   = cupy.ravel( cupy.fromDlpack( self.I.to_dlpack() ) )

        lgm().log(f" --> offsets.ravel = {offsets[0:25]}")
        lgm().log(f" --> I.ravel = {indices[0:25]}")
        lgm().log( f" offsets:   {offsets[0:20]}")
        lgm().log( f" distances: {distances[0:20]}")
        lgm().log( f" indices:   {indices[0:20]}")

        input_cudf = cudf.DataFrame( dict(source=cudf.Series( offsets ), destination=cudf.Series( indices ), distances=cudf.Series( distances ) ) )
        G = cugraph.Graph()
        G.from_cudf_edgelist( input_cudf, source='source', destination='destination', edge_attr='distances', renumber=False )
        self.P = shortest_path( G, source_pid ).sort_values('distance')
        lgm().log(f"Completed spread algorithm in time {time.time() - t0} sec, source = {source_pid}, result {self.P.__class__} = {self.P.head(20)}")
        self.reset = False
        return converged


