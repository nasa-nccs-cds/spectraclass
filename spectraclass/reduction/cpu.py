from __future__ import print_function
import locale
from warnings import warn
import time
from scipy.optimize import curve_fit
from sklearn.utils import check_random_state, check_array
from sklearn.preprocessing import normalize
try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import spectraclass.ext.umap.distances as dist
import spectraclass.ext.umap.sparse as sparse
import numba
from spectraclass.ext.umap.utils import ( tau_rand_int, ts)
from .base import UMAP
from spectraclass.ext.umap.spectral import spectral_layout
from spectraclass.ext.umap.layouts import ( optimize_layout_generic, optimize_layout_inverse )
from spectraclass.ext.pynndescent.distances import named_distances as pynn_named_distances
from spectraclass.ext.pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances
from spectraclass.ext.pynndescent import NNDescent

#from pynndescent import NNDescent
#from pynndescent.distances import named_distances as pynn_named_distances
#from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances
_HAVE_PYNNDESCENT = True

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

@numba.njit()
def clip(val):
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.int64,
    },
)
def rdist(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff
    return result


def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def breadth_first_search(adjmat, start, min_vertices):
    explored = []
    queue = [start]
    levels = {}
    levels[start] = 0
    max_level = np.inf
    visited = [start]

    while queue:
        node = queue.pop(0)
        explored.append(node)
        if max_level == np.inf and len(explored) > min_vertices:
            max_level = max(levels.values())

        if levels[node] + 1 < max_level:
            neighbors = adjmat[node].gindices
            for neighbour in neighbors:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)

                    levels[neighbour] = levels[node] + 1

    return np.array(explored)

@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The gindices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    verbose=False,
):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.

    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. Valid string metrics
        include:
            * euclidean (or l2)
            * manhattan (or l1)
            * cityblock
            * braycurtis
            * canberra
            * chebyshev
            * correlation
            * cosine
            * dice
            * hamming
            * jaccard
            * kulsinski
            * ll_dirichlet
            * mahalanobis
            * matching
            * minkowski
            * rogerstanimoto
            * russellrao
            * seuclidean
            * sokalmichener
            * sokalsneath
            * sqeuclidean
            * yule
            * wminkowski

        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the gindices of the k-nearest neighbors as a row for
        each data point.

    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.

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

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """

    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    rows, cols, vals = compute_membership_strengths( knn_indices, knn_dists, sigmas, rhos )

    lgm().log( f"  EMBED: (vshape, (rmax, cmax)) = ({vals.shape}, ({rows.max()}, {cols.max()})), X shape = ({X.shape[0]}, {X.shape[0]})")
    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()

    return result, sigmas, rhos

@numba.njit()
def fast_intersection(rows, cols, values, target, unknown_dist=1.0, far_dist=5.0):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array
        An array of the value of each non-zero in the sparse matrix
        representation.

    target: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist float (optional, default 5.0)
        The distance between unmatched labels.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        if (target[i] == -1) or (target[j] == -1):
            values[nz] *= np.exp(-unknown_dist)
        elif target[i] != target[j]:
            values[nz] *= np.exp(-far_dist)

    return

@numba.jit()
def fast_metric_intersection(
    rows, cols, values, discrete_space, metric, metric_args, scale
):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array of shape
        An array of the values of each non-zero in the sparse matrix
        representation.

    discrete_space: array of shape (n_samples, n_features)
        The vectors of categorical labels to use in the intersection.

    metric: function
        The function used to calculate distance over the target array.

    scale: float
        A scaling to apply to the metric.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        dist = metric(discrete_space[i], discrete_space[j], *metric_args)
        values[nz] *= np.exp(-(scale * dist))

    return

@numba.njit()
def reprocess_row(probabilities, k=15, n_iters=32):
    target = np.log2(k)

    lo = 0.0
    hi = NPY_INFINITY
    mid = 1.0

    for n in range(n_iters):

        psum = 0.0
        for j in range(probabilities.shape[0]):
            psum += pow(probabilities[j], mid)

        if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
            break

        if psum < target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == NPY_INFINITY:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0

    return np.power(probabilities, mid)

@numba.njit()
def reset_local_metrics(simplicial_set_indptr, simplicial_set_data):
    for i in range(simplicial_set_indptr.shape[0] - 1):
        simplicial_set_data[
            simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]
        ] = reprocess_row(
            simplicial_set_data[simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]]
        )
    return


def reset_local_connectivity(simplicial_set, reset_local_metric=False):
    """Reset the local connectivity requirement -- each data sample should
    have complete confidence in at least one 1-simplex in the simplicial set.
    We can enforce this by locally rescaling confidences, and then remerging the
    different local simplicial sets together.

    Parameters
    ----------
    simplicial_set: sparse matrix
        The simplicial set for which to recalculate with respect to local
        connectivity.

    Returns
    -------
    simplicial_set: sparse_matrix
        The recalculated simplicial set, now with the local connectivity
        assumption restored.
    """
    simplicial_set = normalize(simplicial_set, norm="max")
    if reset_local_metric:
        simplicial_set = simplicial_set.tocsr()
        reset_local_metrics(simplicial_set.indptr, simplicial_set.data)
        simplicial_set = simplicial_set.tocoo()
    transpose = simplicial_set.transpose()
    prod_matrix = simplicial_set.multiply(transpose)
    simplicial_set = simplicial_set + transpose - prod_matrix
    simplicial_set.eliminate_zeros()

    return simplicial_set


def discrete_metric_simplicial_set_intersection(
    simplicial_set,
    discrete_space,
    unknown_dist=1.0,
    far_dist=5.0,
    metric=None,
    metric_kws={},
    metric_scale=1.0,
):
    """Combine a fuzzy simplicial set with another fuzzy simplicial set
    generated from discrete metric data using discrete distances. The target
    data is assumed to be categorical label data (a vector of labels),
    and this will update the fuzzy simplicial set to respect that label data.

    TODO: optional category cardinality based weighting of distance

    Parameters
    ----------
    simplicial_set: sparse matrix
        The input fuzzy simplicial set.

    discrete_space: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist: float (optional, default 5.0)
        The distance between unmatched labels.

    metric: str (optional, default None)
        If not None, then use this metric to determine the
        distance between values.

    metric_scale: float (optional, default 1.0)
        If using a custom metric scale the distance values by
        this value -- this controls the weighting of the
        intersection. Larger values weight more toward target.

    Returns
    -------
    simplicial_set: sparse matrix
        The resulting intersected fuzzy simplicial set.
    """
    simplicial_set = simplicial_set.tocoo()

    if metric is not None:
        # We presume target is now a 2d array, with each row being a
        # vector of target info
        if metric in dist.named_distances:
            metric_func = dist.named_distances[metric]
        else:
            raise ValueError("Discrete intersection metric is not recognized")

        fast_metric_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            metric_func,
            tuple(metric_kws.values()),
            metric_scale,
        )
    else:
        fast_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            unknown_dist,
            far_dist,
        )

    simplicial_set.eliminate_zeros()

    return reset_local_connectivity(simplicial_set)


def general_simplicial_set_intersection(simplicial_set1, simplicial_set2, weight):

    result = (simplicial_set1 + simplicial_set2).tocoo()
    left = simplicial_set1.tocsr()
    right = simplicial_set2.tocsr()

    sparse.general_sset_intersection(
        left.indptr,
        left.gindices,
        left.data,
        right.indptr,
        right.gindices,
        right.data,
        result.row,
        result.col,
        result.data,
        weight,
    )

    return result


def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result

def spectral_embedding( data, graph, n_components = 3, sparsify = False ):
    t0 = time.time()
    graph = graph.tocoo()
    graph.sum_duplicates()
    if sparsify:
        n_epochs = 200
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()

    random_state = np.random.RandomState()
    initialisation = spectral_layout( data, graph, n_components, random_state, metric="euclidean" )
    expansion = 10.0 / np.abs(initialisation).max()
    rv = (initialisation * expansion).astype(np.float32)
    print( f"Completed spectral_embedding in {(time.time()-t0)/60.0} min.")
    return rv


def simplicial_set_embedding(
    data,
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    parallel = True,
    verbose=False,
):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by UMAP.

    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.

    n_components: int
        The dimensionality of the euclidean space into which to embed the data.

    initial_alpha: float
        Initial learning rate for the SGD.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    gamma: float
        Weight to apply to negative samples.

    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.

    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or callable
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.

    metric_kwds: dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.

    output_metric: function
        Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.

    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.

    euclidean_output: bool
        Whether to use the faster code specialised for euclidean output metrics

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """
    t0 = time.time()
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    init_embedding = None

    if isinstance(init, str) and init == "random":
        init_embedding = random_state.uniform( low=-10.0, high=10.0, size=(graph.shape[0], n_components) ).astype(np.float32)
        print(f"Completed UMAP random initialization, init shape = {init_embedding.shape}")
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation = spectral_layout(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        spectral_embedding = (initialisation * expansion).astype(np.float32)
        init_embedding = spectral_embedding + random_state.normal(scale=0.0001, size=[graph.shape[0], n_components]).astype( np.float32 )
        print(f"Completed UMAP spectral initialization, init shape = {init_embedding.shape}")
    else:
        init_embedding = np.array(init)
        print(f"Executing UMAP on existing initialization, init shape = {init_embedding.shape}")

        # if len(init_data.shape) == 2:
        #     if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
        #         tree = KDTree(init_data)
        #         dist, ind = tree.query(init_data, k=2)
        #         nndist = np.mean(dist[:, 1])
        #         embedding = init_data + random_state.normal( scale=0.001 * nndist, size=init_data.shape ).astype(np.float32)
        #     else:
        #         embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    embedding = ( 10.0 * ( init_embedding - np.min(init_embedding, 0) ) / (np.max(init_embedding, 0) - np.min(init_embedding, 0)) ).astype(np.float32, order="C")

    t1 = time.time()
    embedding = optimize_layout_euclidean(
        embedding,
        embedding,
        head,
        tail,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma,
        initial_alpha,
        negative_sample_rate,
        parallel = parallel,
        verbose=verbose,
    )
    t2 = time.time()

    print( f"Completed simplicial_set_embedding, times = {(t1-t0)/60.0} {(t2-t1)/60.0} min")

    return init_embedding, embedding

def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The gindices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The gindices of the tails of 1-simplices with non-zero membership.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """
    from spectraclass.gui.pointcloud import  PointCloudManager
    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    optimize_fn = numba.njit( _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel )
    pcm = PointCloudManager.instance()
    if n_epochs == 1:
        pass
    else:
      print( f" >>> Embed n_epochs={n_epochs}, alpha={alpha} ")
      plot_step = 3
      for n in range(n_epochs):
        if (dim==3) and (n>0) and (n%plot_step==0): pcm.update_plot( points=head_embedding )
        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )
   #     print( f" ITER Embed n={n}, alpha={alpha:.4f}, sample={head_embedding[0]}, dist={head_embedding[0]-head_embedding[1000]} ")
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding

@numba.njit()
def init_transform(indices, weights, embedding):
    """Given gindices and weights and an original embeddings
    initialize the positions of new points relative to the
    gindices and weights (of their neighbors in the source data).

    Parameters
    ----------
    indices: array of shape (n_new_samples, n_neighbors)
        The gindices of the neighbors of each new sample

    weights: array of shape (n_new_samples, n_neighbors)
        The membership strengths of associated 1-simplices
        for each of the new samples.

    embedding: array of shape (n_samples, dim)
        The original embedding of the source data.

    Returns
    -------
    new_embedding: array of shape (n_new_samples, dim)
        An initial embedding of the new sample points.
    """
    result = np.zeros((indices.shape[0], embedding.shape[1]), dtype=np.float32)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for d in range(embedding.shape[1]):
                result[i, d] += weights[i, j] * embedding[indices[i, j], d]

    return result


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class cpUMAP(UMAP):

    def embed( self, X: np.ndarray, y: np.ndarray=None, **kwargs ):
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
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
        progress_callback = kwargs.get('progress_callback')

        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if self._embedding_ is not None:
            init = check_array(self._embedding_, dtype=np.float32, accept_sparse=False)
        elif self._init_embedding_ is not None:
            init = check_array(self._init_embedding_, dtype=np.float32, accept_sparse=False)
        else:
            if isinstance(self.init, np.ndarray):
                init = check_array(self.init, dtype=np.float32, accept_sparse=False)
                lgm().log(f"Running umap[{self.n_components}] with init array, input shape = {self._raw_data.shape}")
            else:
                init = self.init
                lgm().log(f"Running umap[{self.n_components}] with init {init}, input shape = {self._raw_data.shape}")
#                lgm().log( "COMPUTING UMAP" )

        self._initial_alpha = self.learning_rate
        self._validate_parameters()

        if self._sparse_data and not X.has_sorted_indices:
            X.sort_indices()

        random_state = check_random_state(self.random_state)
        if self._sparse_data and self.metric in pynn_sparse_named_distances:
            nn_metric = self.metric
        elif not self._sparse_data and self.metric in pynn_named_distances:
            nn_metric = self.metric
        else:
            nn_metric = self._input_distance_func

        flow: ActivationFlow = afm().getActivationFlow()
        self._knn_indices, self._knn_dists = flow.neighbor_graph

        self.graph_, self._sigmas, self._rhos = fuzzy_simplicial_set(
            X,
            self.n_neighbors,
            random_state,
            nn_metric,
            self._metric_kwds,
            self._knn_indices,
            self._knn_dists,
            self.angular_rp_forest,
            self.set_op_mix_ratio,
            self.local_connectivity,
            True,
            self.verbose,
        )

        # Currently not checking if any duplicate points have differing labels
        # Might be worth throwing a warning...
        if y is not None:
            len_X = len(X) if not self._sparse_data else X.shape[0]
            if len_X != len(y):
                raise ValueError(
                    "Length of x = {len_x}, length of y = {len_y}, while it must be equal.".format(
                        len_x=len_X, len_y=len(y)
                    )
                )
            y_ = check_array(y, ensure_2d=False)
            if self.target_metric == "categorical":
                if self.target_weight < 1.0:
                    far_dist = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    far_dist = 1.0e12
                self.graph_ = discrete_metric_simplicial_set_intersection(
                    self.graph_, y_, far_dist=far_dist
                )
            elif self.target_metric in dist.DISCRETE_METRICS:
                if self.target_weight < 1.0:
                    scale = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    scale = 1.0e12

                metric_kws = dist.get_discrete_params(y_, self.target_metric)
                self.graph_ = discrete_metric_simplicial_set_intersection(
                    self.graph_,
                    y_,
                    metric=self.target_metric,
                    metric_kws=metric_kws,
                    metric_scale=scale,
                )
            else:
                if len(y_.shape) == 1:
                    y_ = y_.reshape(-1, 1)
                if self.target_n_neighbors == -1:
                    target_n_neighbors = self.n_neighbors
                else:
                    target_n_neighbors = self.target_n_neighbors

                target_graph, target_sigmas, target_rhos = fuzzy_simplicial_set(
                    y_,
                    target_n_neighbors,
                    random_state,
                    self.target_metric,
                    self._target_metric_kwds,
                    None,
                    None,
                    False,
                    1.0,
                    1.0,
                    False,
                )

                self.graph_ = general_simplicial_set_intersection( self.graph_, target_graph, self.target_weight )
                self.graph_ = reset_local_connectivity(self.graph_)

        if self.verbose:
            print(ts(), "Construct embedding")

        nepochs = kwargs.get( 'nepochs', self.n_epochs )
        init_alpha = kwargs.get( 'alpha', self._initial_alpha )
        t0 = time.time()
        print( f"Computing umap embedding with nepochs = {nepochs}, alpha = {init_alpha}, nLabels = {np.count_nonzero( y > 0 )}, n_neighbors = {self.n_neighbors}, init = {init} ({self.init})" )
        self._init_embedding_, self._embedding_ = simplicial_set_embedding(
            self._raw_data,  # JH why raw data?
            self.graph_,
            self.n_components,
            init_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            nepochs,
            init,
            random_state,
            self._input_distance_func,
            self._metric_kwds,
            self.parallel,
            self.verbose,
        )
        print(f"umap embedding complete, result shape = {self._embedding_.shape}, time = {time.time()-t0} " )

        self._input_hash = joblib.hash(self._raw_data)
        return self


    def transform(self, X):
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
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
        # If we fit just a single instance then error
        if self.embedding.shape[0] == 1:
            raise ValueError(
                "Transform unavailable when model was fit with only a single data sample."
            )
        # If we just have the original input then short circuit things
        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        x_hash = joblib.hash(X)
        if x_hash == self._input_hash:
            return self.embedding

        if self.metric == "precomputed":
            raise ValueError(
                "Transform  of new data not available for precomputed metric."
            )

        # X = check_array(X, dtype=np.float32, order="C", accept_sparse="csr")
        random_state = check_random_state(self.transform_seed)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        flow: ActivationFlow = afm().getActivationFlow()
        indices, dists = flow.query(X, self.n_neighbors)
        dists = dists.astype(np.float32, order="C")

        adjusted_local_connectivity = max(0.0, self.local_connectivity - 1.0)
        sigmas, rhos = smooth_knn_dist( dists, float(self.n_neighbors), local_connectivity=float(adjusted_local_connectivity) )
        rows, cols, vals = compute_membership_strengths(indices, dists, sigmas, rhos)
        graph = scipy.sparse.coo_matrix( (vals, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])  )

        # This was a very specially constructed graph with constant degree.
        # That lets us do fancy unpacking by reshaping the csr matrix gindices
        # and data. Doing so relies on the constant degree assumption!
        csr_graph = normalize(graph.tocsr(), norm="l1")
        inds = csr_graph.gindices.reshape(X.shape[0], self.n_neighbors)
        weights = csr_graph.data.reshape(X.shape[0], self.n_neighbors)
        embedding = init_transform(inds, weights, self.embedding)

        if self.n_epochs == 0:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 100
            else:
                n_epochs = 30
        else:
            n_epochs = int(self.n_epochs // 3.0)

        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col
        weight = graph.data

        # optimize_layout = make_optimize_layout(
        #     self._output_distance_func,
        #     tuple(self.output_metric_kwds.values()),
        # )

        if self.output_metric == "euclidean":
            embedding = optimize_layout_euclidean(
                embedding,
                self.embedding.astype(np.float32, copy=True),  # Fixes #179 & #217,
                head,
                tail,
                n_epochs,
                graph.shape[1],
                epochs_per_sample,
                self._a,
                self._b,
                rng_state,
                self.repulsion_strength,
                self._initial_alpha / 4.0,
                self.negative_sample_rate,
                self.random_state is None,
                verbose=self.verbose,
            )
        else:
            embedding = optimize_layout_generic(
                embedding,
                self.embedding.astype(np.float32, copy=True),  # Fixes #179 & #217
                head,
                tail,
                n_epochs,
                graph.shape[1],
                epochs_per_sample,
                self._a,
                self._b,
                rng_state,
                self.repulsion_strength,
                self._initial_alpha / 4.0,
                self.negative_sample_rate,
                self._output_distance_func,
                tuple(self._output_metric_kwds.values()),
                verbose=self.verbose,
            )

        return embedding

    def inverse_transform(self, X):
        """Transform X in the existing embedded space back into the input
        data space and return that transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_components)
            New points to be inverse transformed.

        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Generated data points new data in data space.
        """

        if self._sparse_data:
            raise ValueError("Inverse transform not available for sparse input.")
        elif self._inverse_distance_func is None:
            raise ValueError("Inverse transform not available for given metric.")
        elif self.n_components >= 8:
            warn(
                "Inverse transform works best with low dimensional embeddings."
                " Results may be poor, or this approach to inverse transform"
                " may fail altogether! If you need a high dimensional latent"
                " space and inverse transform operations consider using an"
                " autoencoder."
            )

        X = check_array(X, dtype=np.float32, order="C")
        random_state = check_random_state(self.transform_seed)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        # build Delaunay complex (Does this not assume a roughly euclidean output metric)?
        deltri = scipy.spatial.Delaunay(
            self.embedding, incremental=True, qhull_options="QJ"
        )
        neighbors = deltri.simplices[deltri.find_simplex(X)]
        adjmat = scipy.sparse.lil_matrix(
            (self.embedding.shape[0], self.embedding.shape[0]), dtype=int
        )
        for i in np.arange(0, deltri.simplices.shape[0]):
            for j in deltri.simplices[i]:
                if j < self.embedding.shape[0]:
                    idx = deltri.simplices[i][
                        deltri.simplices[i] < self.embedding.shape[0]
                    ]
                    adjmat[j, idx] = 1
                    adjmat[idx, j] = 1

        adjmat = scipy.sparse.csr_matrix(adjmat)

        min_vertices = min(self._raw_data.shape[-1], self._raw_data.shape[0])

        neighborhood = [
            breadth_first_search(adjmat, v[0], min_vertices=min_vertices)
            for v in neighbors
        ]
        if callable(self.output_metric):
            _out_m = self.output_metric

            @numba.njit(fastmath=True)
            def _output_dist_only(x, y, *kwds):
                return _out_m(x, y, *kwds)[0]
            dist_only_func = _output_dist_only
        elif self.output_metric in dist.named_distances.keys():
            dist_only_func = dist.named_distances[self.output_metric]
        else:
            # shouldn't really ever get here because of checks already performed,
            # but works as a failsafe in case attr was altered manually after fitting
            raise ValueError(
                "Unrecognized output metric: {}".format(self.output_metric)
            )

        dist_args = tuple(self._output_metric_kwds.values())
        distances = [
            np.array(
                [
                    dist_only_func(X[i], self.embedding[nb], *dist_args)
                    for nb in neighborhood[i]
                ]
            )
            for i in range(X.shape[0])
        ]
        idx = np.array([np.argsort(e)[:min_vertices] for e in distances])

        dists_output_space = np.array(
            [distances[i][idx[i]] for i in range(len(distances))]
        )
        indices = np.array([neighborhood[i][idx[i]] for i in range(len(neighborhood))])

        rows, cols, distances = np.array(
            [
                [i, indices[i, j], dists_output_space[i, j]]
                for i in range(indices.shape[0])
                for j in range(min_vertices)
            ]
        ).T

        # calculate membership strength of each edge
        weights = 1 / (1 + self._a * distances ** (2 * self._b))

        # compute 1-skeleton
        # convert 1-skeleton into coo_matrix adjacency matrix
        graph = scipy.sparse.coo_matrix(
            (weights, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])
        )

        # That lets us do fancy unpacking by reshaping the csr matrix gindices
        # and data. Doing so relies on the constant degree assumption!
        # csr_graph = graph.tocsr()
        csr_graph = normalize(graph.tocsr(), norm="l1")
        inds = csr_graph.gindices.reshape(X.shape[0], min_vertices)
        weights = csr_graph.data.reshape(X.shape[0], min_vertices)
        inv_transformed_points = init_transform(inds, weights, self._raw_data)

        if self.n_epochs == 0:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 100
            else:
                n_epochs = 30
        else:
            n_epochs = int(self.n_epochs // 3.0)

        # graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        # graph.eliminate_zeros()

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col
        weight = graph.data

        inv_transformed_points = optimize_layout_inverse(
            inv_transformed_points,
            self._raw_data,
            head,
            tail,
            weight,
            self._sigmas,
            self._rhos,
            n_epochs,
            graph.shape[1],
            epochs_per_sample,
            self._a,
            self._b,
            rng_state,
            self.repulsion_strength,
            self._initial_alpha / 4.0,
            self.negative_sample_rate,
            self._inverse_distance_func,
            tuple(self._metric_kwds.values()),
            verbose=self.verbose,
        )

        return inv_transformed_points