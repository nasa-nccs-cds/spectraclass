import os, time, random, numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.axes import Axes, BarContainer
from spectraclass.data.base import DataManager, dm
import xarray as xa
TEST_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from pynndescent import NNDescent
from sklearn.decomposition import PCA, FastICA
from pywt import dwt2

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def center( x: np.ndarray, axis = 0 ) -> np.ndarray:
    return ( x - x.mean(axis=axis) ) / x.std(axis=axis)

def norm( x: np.ndarray, axis = 0 ) -> np.ndarray:
    return x / x.std(axis=axis)

def load_test_data( dataset_type: str, dsid: str, data_type: str, ilayer: int, block: List[int] = None ) -> xa.DataArray:
    if block is None:
        data_file = os.path.join(TEST_DIR, "data", dataset_type, f"{dsid}_{data_type}_{ilayer}.nc4")
    else:
        data_file = os.path.join( dm().cache_dir, "test", "data", dataset_type, f"ks_raw_{block[0]}_{block[1]}_{ilayer}.nc4")
    dataset = xa.load_dataset(data_file)
    return dataset['data']

def plot( axs: np.ndarray, iP: int, data: np.ndarray, title: str, cmap: str = 'jet' ) -> AxesImage:
    if axs.ndim == 2:
        ncol: int = axs.shape[-1]
        return plot2( axs, iP//ncol, iP%ncol, data, title, cmap )
    else:
        return plot2( axs, iP, -1, data, title, cmap )

def plot2( axs: np.ndarray, i0: int, i1: int, data: np.ndarray, title: str, cmap: str = 'jet' ) -> AxesImage:      # 'nipy_spectral'
    ax: Axes  = axs[ i0, i1 ] if i1 >= 0 else axs[i0]
    ax.set_yticks([]); ax.set_xticks([])
    ax.title.set_text( title )
    imgplot: AxesImage = ax.imshow(data)
    if data.ndim == 2:  imgplot.set_cmap( cmap )
    return imgplot

def getProbabilityGraph( data: np.ndarray, nneighbors: int ) -> NNDescent:    # data: array, shape = (n_samples, n_features)
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, 2 * int(round(np.log2(data.shape[0]))))
    kwargs = dict(n_trees=n_trees, n_iters=n_iters, n_neighbors=nneighbors, max_candidates=60, verbose=True, metric="hellinger" )
    nnd =  NNDescent(data, **kwargs)
    return nnd

def ca_reduction( data: np.ndarray, ndim: int, method: str = "pca"  ) -> Tuple[np.ndarray,np.ndarray]:
    if method == "pca":
        mapper = PCA(n_components=ndim)
    elif method == "ica":
        mapper = FastICA(n_components=ndim)
    else:
        raise Exception(f"Unknown reduction methos: {method}")
    centered_data: np.ndarray = center(data)
    mapper.fit( centered_data )
    if method == "pca":
       print(f"PCA reduction[{ndim}], Percent variance explained: {mapper.explained_variance_ratio_ * 100}")

    reduced_features: np.ndarray = mapper.transform( centered_data )
    reproduction: np.ndarray = mapper.inverse_transform( reduced_features )
    return  ( reduced_features, reproduction )


def apply_standard_pca( array: np.ndarray, n_components: int ):
    pca = PCA(n_components=n_components)
    pca.fit(array)
    print(f"PCA reduction[{n_components}], Percent variance explained: {pca.explained_variance_ratio_ * 100}")
    transformed_data = pca.transform(array)
    return transformed_data

def autoencoder_reduction( input_data: np.ndarray, ndim: int, epochs: int = 70, **kwargs )-> Tuple[np.ndarray,np.ndarray]:  #  input_data:  [ n_features, n_samples ]
    from keras.layers import Input, Dense
    from keras.models import Model

    activation = kwargs.get( 'activation', 'tanh' )
    optimizer = kwargs.get( 'optimizer', 'rmsprop')
    loss = kwargs.get( 'loss', "mean_squared_error" )

    input_dims = input_data.shape[1]
    reduction_factor = 2
    inputlayer = Input( shape=[input_dims] )
    layer_dims, x = int( round( input_dims / reduction_factor )), inputlayer
    while layer_dims > ndim:
        x = Dense(layer_dims, activation=activation)(x)
        layer_dims = int( round( layer_dims / reduction_factor ))
    encoded = x = Dense( ndim, activation=activation )(x)
    layer_dims = int( round( ndim * reduction_factor ))
    while layer_dims < input_dims:
        x = Dense(layer_dims, activation=activation)(x)
        layer_dims = int( round( layer_dims * reduction_factor ))
    decoded = Dense( input_dims, activation='sigmoid' )(x)
    autoencoder = Model(inputs=[inputlayer], outputs=[decoded])
    encoder = Model(inputs=[inputlayer], outputs=[encoded])
    autoencoder.summary()
    encoder.summary()

    autoencoder.compile(loss=loss, optimizer=optimizer )
    autoencoder.fit( input_data, input_data, epochs=epochs, batch_size=256, shuffle=True )
    encoded_data: np.ndarray = encoder.predict( input_data )
    reproduction: np.ndarray = autoencoder.predict( input_data )
    print(f" Autoencoder_reduction, result: shape = {encoded_data.shape}")
    print(f" ----> encoder_input: shape = {input_data.shape}, range = {[input_data.min(),input_data.max()]} ")
    print(f" ----> reproduction: shape = {reproduction.shape}, range = {[reproduction.min(),reproduction.max()]} ")
    print(f" ----> encoding: shape = {encoded_data.shape}, range = {[encoded_data.min(),encoded_data.max()]} ")
    return ( encoded_data, reproduction )

def getEnergyDensity( image: np.ndarray ) -> np.ndarray:
    _, (cH, cV, cD) = dwt2( image, 'db1' )
    return ( cH ** 2 + cV ** 2 + cD ** 2 )

def getImageEnergy( image: np.ndarray ) -> float:
    energy_density = getEnergyDensity( image )
    return energy_density.sum() / image.size

def get_magnitude(response):
    magnitude = np.array([np.sqrt(response[0][i][j]**2+response[1][i][j]**2)
                        for i in range(len(response[0])) for j in range(len(response[0][i]))])
    return magnitude

def get_image_energy(pixels):
    _, (cH, cV, cD) = dwt2(pixels.T, 'db1')
    energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / pixels.size
    return energy

def get_energy_density(pixels):
    energy = get_image_energy(pixels)
    energy_density = energy / (pixels.shape[0]*pixels.shape[1])
    return round(energy_density*100,5)



def _get_chunks(shape, ncpu):
    """Split the array into equal sized chunks based on the number of
    available processors. The last chunk in each dimension absorbs the
    remainder array elements if the number of CPUs does not divide evenly into
    the number of array elements.

    Examples
    --------
    >>> _get_chunks((4, 4), 4)
    ((2, 2), (2, 2))
    >>> _get_chunks((4, 4), 2)
    ((2, 2), (4,))
    >>> _get_chunks((5, 5), 2)
    ((2, 3), (5,))
    >>> _get_chunks((2, 4), 2)
    ((1, 1), (4,))
    """
    # since apply_parallel is in the critical import path, we lazy import
    # math just when we need it.
    from math import ceil

    chunks = []
    nchunks_per_dim = int(ceil(ncpu ** (1./len(shape))))

    used_chunks = 1
    for i in shape:
        if used_chunks < ncpu:
            regular_chunk = i // nchunks_per_dim
            remainder_chunk = regular_chunk + (i % nchunks_per_dim)

            if regular_chunk == 0:
                chunk_lens = (remainder_chunk,)
            else:
                chunk_lens = ((regular_chunk,) * (nchunks_per_dim - 1) +
                              (remainder_chunk,))
        else:
            chunk_lens = (i,)

        chunks.append(chunk_lens)
        used_chunks *= nchunks_per_dim
    return tuple(chunks)


def _ensure_dask_array(array, chunks=None):
    import dask.array as da
    if isinstance(array, da.Array):
        return array

    return da.from_array(array, chunks=chunks)


def apply_parallel(function, array, chunks=None, depth=0, mode=None,
                   extra_arguments=(), extra_keywords={}, *, compute=None):
    """Map a function in parallel across an array.

    Split an array into possibly overlapping chunks of a given depth and
    boundary type, call the given function in parallel on the chunks, combine
    the chunks and return the resulting array.

    Parameters
    ----------
    function : function
        Function to be mapped which takes an array as an argument.
    array : numpy array or dask array
        Array which the function will be applied to.
    chunks : int, tuple, or tuple of tuples, optional
        A single integer is interpreted as the length of one side of a square
        chunk that should be tiled across the array.  One tuple of length
        ``array.ndim`` represents the shape of a chunk, and it is tiled across
        the array.  A list of tuples of length ``ndim``, where each sub-tuple
        is a sequence of chunk sizes along the corresponding dimension. If
        None, the array is broken up into chunks based on the number of
        available cpus. More information about chunks is in the documentation
        `here <https://dask.pydata.org/en/latest/array-design.html>`_.
    depth : int, optional
        Integer equal to the depth of the added boundary cells. Defaults to
        zero.
    mode : {'reflect', 'symmetric', 'periodic', 'wrap', 'nearest', 'edge'}, optional
        type of external boundary padding.
    extra_arguments : tuple, optional
        Tuple of arguments to be passed to the function.
    extra_keywords : dictionary, optional
        Dictionary of keyword arguments to be passed to the function.
    compute : bool, optional
        If ``True``, compute eagerly returning a NumPy Array.
        If ``False``, compute lazily returning a Dask Array.
        If ``None`` (default), compute based on array type provided
        (eagerly for NumPy Arrays and lazily for Dask Arrays).

    Returns
    -------
    out : ndarray or dask Array
        Returns the result of the applying the operation.
        Type is dependent on the ``compute`` argument.

    Notes
    -----
    Numpy edge modes 'symmetric', 'wrap', and 'edge' are converted to the
    equivalent ``dask`` boundary modes 'reflect', 'periodic' and 'nearest',
    respectively.
    Setting ``compute=False`` can be useful for chaining later operations.
    For example region selection to preview a result or storing large data
    to disk instead of loading in memory.

    """
    try:
        import dask.array as da
    except ImportError:
        raise RuntimeError("Could not import 'dask'.  Please install using 'pip install dask'")

    if compute is None:
        compute = not isinstance(array, da.Array)

    if chunks is None:
        shape = array.shape
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
        except NotImplementedError:
            ncpu = 4
        chunks = _get_chunks(shape, ncpu)

    if mode == 'wrap':          mode = 'periodic'
    elif mode == 'symmetric':   mode = 'reflect'
    elif mode == 'edge':        mode = 'nearest'

    def wrapped_func(arr):
        return function(arr, *extra_arguments, **extra_keywords)

    darr = _ensure_dask_array(array, chunks=chunks)

    res = darr.map_overlap( wrapped_func, depth, boundary=mode )
    if compute:
        res = res.compute()

    return res


