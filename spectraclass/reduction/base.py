from __future__ import print_function
import locale
from warnings import warn
from sklearn.base import BaseEstimator
import numpy as np
import xarray as xa
import scipy.sparse
import scipy.sparse.csgraph
import numba
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

class UMAP(BaseEstimator):
    """Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided.  Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * ll_dirichlet
            * hellinger
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.

    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.

    low_memory: bool (optional, default False)
        For some datasets the nearest neighbor computation can consume a lot of
        memory. If you find that UMAP is failing due to memory constraints
        consider setting this option to True. This approach is more
        computationally expensive, but avoids excessive memory use.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.

    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model_
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.

    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    metric_kwds: dict (optional, default None)
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. If None then no arguments are passed on.

    angular_rp_forest: bool (optional, default False)
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search. This can be faster, but is
        mostly on useful for metric that use an angular style distance such
        as cosine, correlation etc. In the case of those metrics angular forests
        will be chosen automatically.

    target_n_neighbors: int (optional, default -1)
        The number of nearest neighbors to use to construct the target simplcial
        set. If set to -1 use the ``n_neighbors`` value.

    target_metric: string or callable (optional, default 'categorical')
        The metric used to measure distance for a target array is using supervised
        dimension reduction. By default this is 'categorical' which will measure
        distance in terms of whether categories match or are different. Furthermore,
        if semi-supervised is required target values of -1 will be trated as
        unlabelled under the 'categorical' metric. If the target array takes
        continuous values (e.g. for a regression problem) then metric of 'l1'
        or 'l2' is probably more appropriate.

    target_metric_kwds: dict (optional, default None)
        Keyword argument to pass to the target metric when performing
        supervised dimension reduction. If None then no arguments are passed on.

    target_weight: float (optional, default 0.5)
        weighting factor between data topology and target topology. A value of
        0.0 weights entirely on data, a value of 1.0 weights entirely on target.
        The default of 0.5 balances the weighting equally between data and target.

    transform_seed: int (optional, default 42)
        Random seed used for the stochastic aspects of the transform operation.
        This ensures consistency in transform operations.

    verbose: bool (optional, default False)
        Controls verbosity of logging.

    unique: bool (optional, default False)
        Controls if the rows of your data should be uniqued before being
        embedded.  If you have more duplicates than you have n_neighbour
        you can have the identical data points lying in different regions of
        your space.  It also violates the definition of a metric.
    """

    def __init__(
        self,
        n_neighbors=10,
        n_components=2,
        metric="euclidean",
        metric_kwds=None,
        output_metric="euclidean",
        output_metric_kwds=None,
        n_epochs=50,
        learning_rate=1.0,
        init="random",
        min_dist=0.1,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        force_approximation_algorithm=False,
        parallel = True,
        verbose=False,
        unique=False,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.output_metric = output_metric
        self.target_metric = target_metric
        self.metric_kwds = metric_kwds
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init if init is not None else "random"
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate
        self._embedding_ = None
        self._init_embedding_ = None
        self.external_embedding = None

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.parallel = parallel
        self.verbose = verbose
        self.unique = unique
        self._raw_data = None

        self.a = a
        self.b = b

        self._input_data: np.ndarray = None
        self.scoord: xa.DataArray = None

    @property
    def input_data(self) -> np.ndarray:
        from spectraclass.gui.spatial.map import MapManager, mm
#        from spectraclass.data.spatial.tile.manager import TileManager, tm
#        result, mask = mm().block.raster2points( mm().threshold_mask )
        return self._input_data

    @input_data.setter
    def input_data(self, value):
        self._input_data = value

    def getNNGraph( self ):
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        return afm().getActivationFlow().getGraph()

        # n_trees = kwargs.get('ntree', 5 + int(round((nodes.shape[0]) ** 0.5 / 20.0)))
        # n_iters = kwargs.get('niter', max(5, 2 * int(round(np.log2(nodes.shape[0])))))
        # nnd = NNDescent(nodes, n_trees=n_trees, n_iters=n_iters, n_neighbors=self.n_neighbors, max_candidates=60, verbose=True)
        # return nnd

    def set_embedding(self, embed_ : np.ndarray ):
        self.external_embedding = embed_

    def init_embedding(self, embed_ : np.ndarray ):
        self._init_embedding_ = embed_

    @property
    def initial( self ) ->  np.ndarray:
        return self._init_embedding_

    def clear_initialization( self ):
        self._init_embedding_ = None

    def clear_embedding( self ):
        self._embedding_ = None
        self.external_embedding = None

    @property
    def embedding( self ) ->  np.ndarray:
        if self.external_embedding is not None: return self.external_embedding
        if self._embedding_ is not None: return self._embedding_
        return self._init_embedding_

    def _validate_parameters(self):
        import spectraclass.ext.umap.distances as dist
        import spectraclass.ext.umap.sparse as sparse
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if self._init_embedding_ is None:
            if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
                raise ValueError(f"init must be a string or ndarray: {self.init}")
            if isinstance(self.init, str) and self.init not in ("spectral", "random"):
                raise ValueError('string init values must be "spectral" or "random"')
            if ( isinstance(self.init, np.ndarray) and self.init.shape[1] != self.n_components ):
                raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.metric_kwds is None:
            self._metric_kwds = {}
        else:
            self._metric_kwds = self.metric_kwds
        if self.output_metric_kwds is None:
            self._output_metric_kwds = {}
        else:
            self._output_metric_kwds = self.output_metric_kwds
        if self.target_metric_kwds is None:
            self._target_metric_kwds = {}
        else:
            self._target_metric_kwds = self.target_metric_kwds
        # check sparsity of data upfront to set proper _input_distance_func &
        # save repeated checks later on
        if scipy.sparse.isspmatrix_csr(self._raw_data):
            self._sparse_data = True
        else:
            self._sparse_data = False
        # set input distance metric & inverse_transform distance metric
        if callable(self.metric):
            in_returns_grad = self._check_custom_metric(
                self.metric, self._metric_kwds, self._raw_data
            )
            if in_returns_grad:
                _m = self.metric

                @numba.njit(fastmath=True)
                def _dist_only(x, y, *kwds):
                    return _m(x, y, *kwds)[0]

                self._input_distance_func = _dist_only
                self._inverse_distance_func = self.metric
            else:
                self._input_distance_func = self.metric
                self._inverse_distance_func = None
                warn(
                    "custom distance metric does not return gradient; inverse_transform will be unavailable. "
                    "To enable using inverse_transform method method, define a distance function that returns "
                    "a tuple of (distance [float], gradient [np.array])"
                )
        elif self.metric == "precomputed":
            if self.unique is False:
                raise ValueError("unique is poorly defined on a precomputed metric")
            warn(
                "using precomputed metric; transform will be unavailable for new data and inverse_transform "
                "will be unavailable for all data"
            )
            self._input_distance_func = self.metric
            self._inverse_distance_func = None
        elif self.metric == "hellinger" and self._raw_data.min() < 0:
            raise ValueError("Metric 'hellinger' does not support negative values")
        elif self.metric in dist.named_distances:
            if self._sparse_data:
                if self.metric in sparse.sparse_named_distances:
                    self._input_distance_func = sparse.sparse_named_distances[
                        self.metric
                    ]
                else:
                    raise ValueError(
                        "Metric {} is not supported for sparse data".format(self.metric)
                    )
            else:
                self._input_distance_func = dist.named_distances[self.metric]
            try:
                self._inverse_distance_func = dist.named_distances_with_gradients[
                    self.metric
                ]
            except KeyError:
                warn(
                    "gradient function is not yet implemented for {} distance metric; "
                    "inverse_transform will be unavailable".format(self.metric)
                )
                self._inverse_distance_func = None
        else:
            raise ValueError("metric is neither callable nor a recognised string")
        # set ooutput distance metric
        if callable(self.output_metric):
            out_returns_grad = self._check_custom_metric(
                self.output_metric, self._output_metric_kwds
            )
            if out_returns_grad:
                self._output_distance_func = self.output_metric
            else:
                raise ValueError(
                    "custom output_metric must return a tuple of (distance [float], gradient [np.array])"
                )
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        elif self.output_metric in dist.named_distances_with_gradients:
            self._output_distance_func = dist.named_distances_with_gradients[
                self.output_metric
            ]
        elif self.output_metric in dist.named_distances:
            raise ValueError(
                "gradient function is not yet implemented for {}.".format(
                    self.output_metric
                )
            )
        else:
            raise ValueError(
                "output_metric is neither callable nor a recognised string"
            )
        # set angularity for NN search based on metric
        if self.metric in (
            "cosine",
            "correlation",
            "dice",
            "jaccard",
            "ll_dirichlet",
            "hellinger",
        ):
            self.angular_rp_forest = True

    def _check_custom_metric(self, metric, kwds, data=None):
        # quickly check to determine whether user-defined
        # self.metric/self.output_metric returns both distance and gradient
        if data is not None:
            # if checking the high-dimensional distance metric, test directly on
            # input data so we don't risk violating any assumptions potentially
            # hard-coded in the metric (e.g., bounded; non-negative)
            x, y = data[np.random.randint(0, data.shape[0], 2)]
        else:
            # if checking the manifold distance metric, simulate some data on a
            # reasonable interval with output dimensionality
            x, y = np.random.uniform(low=-10, high=10, size=(2, self.n_components))
        metric_out = metric(x, y, **kwds)
        # True if metric returns iterable of length 2, False otherwise
        return hasattr(metric_out, "__iter__") and len(metric_out) == 2

    @log_timing
    def embed( self, X: np.ndarray, y: np.ndarray=None, **kwargs ):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        """
        raise NotImplementedError()

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        raise NotImplementedError()

    @classmethod
    def instance( cls, *args, **kwargs ):
        from spectraclass.data.base import DataManager
        ptype = DataManager.instance().proc_type
        mapper = None
        if ptype == "cpu":
            from .cpu import cpUMAP
            mapper = cpUMAP(*args, **kwargs)
        elif ptype == "gpu":
            from .gpu import gpUMAP
            mapper = gpUMAP(*args, **kwargs)
        else:
            print(f"Error, unknown proc_type: {ptype}")
        return mapper

