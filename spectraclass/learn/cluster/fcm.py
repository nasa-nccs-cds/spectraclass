from typing import Optional

import numpy as np
from numpy.typing import NDArray
import xarray as xa
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from .base import ClusterBase

class FCM(ClusterBase):
    r"""Fuzzy C-means Model
    Attributes:
        n_clusters (int): The number of clusters to form as well as the number
        of centroids to generate by the fuzzy C-means.
        max_iter (int): Maximum number of iterations of the fuzzy C-means
        algorithm for a single run.
        m (float): Degree of fuzziness: $m \in (1, \infty)$.
        error (float): Relative tolerance with regards to Frobenius norm of
        the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        random_state (Optional[int]): Determines random number generation for
        centroid initialization.
        Use an int to make the randomness deterministic.
        trained (bool): Variable to store whether or not the model has been
        trained.
    Returns: FCM: A FCM model.
    Raises: ReferenceError: If called without the model being trained
    """

    def __init__( self, n_clusters: int, **kwargs ):
        ClusterBase.__init__( self, n_clusters )
        self.random_state = kwargs.get( 'random_state', 100 )
        self.max_iter = kwargs.get('max_iter', 150 )
        self.m = kwargs.get('m', 2.0 )
        self.error: float = kwargs.get( 'error', 1e-5 )
        self._threshold = 0.0
        self.threshold_mask = None
        self.trained: bool = False
        self._samples = None
        self._attrs = None
        self._fuzzy_cluster_data = None

    def reset(self):
        self._threshold = 0.0
        self.threshold_mask = None
        self.trained: bool =False
        self._samples = None
        self._attrs = None
        self._fuzzy_cluster_data = None
        lgm().log(f"FCM.cluster: reset")

    def _update_nclusters( self ):
        self._fuzzy_cluster_data = None
        lgm().log(f"FCM.cluster: update_nclusters")

    @property
    def cluster_data(self) -> xa.DataArray:
        cdata = np.expand_dims(np.argmax(self._fuzzy_cluster_data * self.cscale, axis=1), 1) + 1
        if self._threshold > 0.0: cdata[ self.threshold_mask ] = 0
        crange = [cdata.min(),cdata.max()]
        lgm().log( f"FCM.cluster_data: shape = {cdata.shape}, thresh = {self._threshold}, range = [{self._fuzzy_cluster_data.min()},{self._fuzzy_cluster_data.max()}]")
        for iC in range( crange[0], crange[1]+1 ):
            lgm().log(f" --> CLASS {iC}: npix= {np.count_nonzero(cdata==iC)}")
        return xa.DataArray(cdata, dims=['samples', 'clusters'], name="clusters", coords=dict(samples=self._samples, clusters=[0]), attrs=self._attrs)

    def fuzzy_cluster_data(self) -> xa.DataArray:
        return self._fuzzy_cluster_data

    def cluster( self, data: xa.DataArray, y=None ):
        self._attrs = data.attrs
        self._samples = data.coords[ data.dims[0] ]
        self.fit( data.values )
        self._fuzzy_cluster_data = self.soft_predict(data.values)
        lgm().log(  f"FCM.cluster: data shape = {data.shape}, data range = [{data.values.min()},{data.values.max()}], "
                    f"clusters shape = {self._fuzzy_cluster_data.shape}, range = [{self._fuzzy_cluster_data.min()},{self._fuzzy_cluster_data.max()}]")

    def threshold( self, thresh: float ):
        self._threshold = thresh
        if self._fuzzy_cluster_data is not None:
            cluster_data_peak = self._fuzzy_cluster_data.max(axis=1)
            self.threshold_mask = cluster_data_peak < self._threshold
            lgm().log(f"FCM.threshold: cluster data shape = {self._fuzzy_cluster_data.shape}, #gtz = {np.count_nonzero(self.threshold_mask)} ")

    def fit(self, X: NDArray) -> None:
        self.rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
        self.u = self.u / np.tile( self.u.sum(axis=1)[np.newaxis].T, self.n_clusters )
        for i in range(self.max_iter):
            u_old = self.u.copy()
            self._centers = FCM._next_centers(X, self.u, self.m)
            self.u = self.soft_predict(X)
            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.trained = True

    def soft_predict(self, X: NDArray) -> NDArray:
        """Soft predict of FCM
        Args:  X (NDArray): New data to predict.
        Returns:
            NDArray: Fuzzy partition array, returned as an array with
            n_samples rows and n_clusters columns.
        """
        temp = FCM._dist(X, self._centers) ** (2 / (self.m - 1))
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat( temp.shape[-1], axis=1 )
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)

    def predict(self, X: NDArray) -> NDArray:
        """Predict the closest cluster each sample in X belongs to.
        Args: X (NDArray): New data to predict.
        Raises: ReferenceError: If it called without the model being trained.
        Returns: NDArray: Index of the cluster each sample belongs to.
        """
        if self._is_trained():
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
            fuzzy_result = self.soft_predict(X)
            lgm().log( f"FCM: fuzzy_result range = [{fuzzy_result.min()},{fuzzy_result.max()}]")
            return fuzzy_result.argmax(axis=-1)
        raise ReferenceError( "You need to train the model. Run `.fit()` method to this." )

    def _is_trained(self) -> bool:
        if self.trained:
            return True
        return False

    @staticmethod
    def _dist(A: NDArray, B: NDArray) -> NDArray:
        """Compute the euclidean distance two matrices"""
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

    @staticmethod
    def _next_centers(X: NDArray, u: NDArray, m: float):
        """Update cluster centers"""
        um = u**m
        return (X.T @ um / np.sum(um, axis=0)).T

    @property
    def centers(self) -> NDArray:
        if self._is_trained():
            return self._centers
        raise ReferenceError( "You need to train the model. Run `.fit()` method to this." )

    @property
    def partition_coefficient(self) -> float:
        """Partition coefficient

        Equation 12a of
        [this paper](https://doi.org/10.1016/0098-3004(84)90020-7).
        """
        if self._is_trained():
            return np.mean(self.u**2)
        raise ReferenceError( "You need to train the model. Run `.fit()` method to this." )

    @property
    def partition_entropy_coefficient(self):
        if self._is_trained():
            return -np.mean(self.u * np.log2(self.u))
        raise ReferenceError( "You need to train the model. Run `.fit()` method to this." )
