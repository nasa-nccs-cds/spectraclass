from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Union, Tuple, Optional, Dict, Callable
import os, numpy as np, xarray as xa
# from pynndescent import NNDescent
from spectraclass.ext.pynndescent import NNDescent

dm: DataManager = DataManager.initialize("img_mgr", 'aviris')
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"

block_size = 150
method = "vae"
model_dims = 16

dm.modal.ext = "_img"
dm.use_model_data = True
TileManager.block_size = block_size
TileManager.block_index = [0, 7]
AvirisDataManager.valid_aviris_bands = [[5, 193], [214, 283], [319, 10000]]
AvirisDataManager.model_dims = model_dims
AvirisDataManager.reduce_method = method
AvirisDataManager.modelkey = f"b{block_size}.{method}"

dm.loadCurrentProject()
classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

lm().setLabels( classes )
dm.modal.initialize_dimension_reduction()
block = tm().getBlock()

nneighbors = 5
metric =  'euclidean'
p = 2

nodes: xa.DataArray = dm.getModelData()
n_trees = 5 + int(round((nodes.shape[0]) ** 0.5 / 20.0))
n_iters = max(5, 2 * int(round(np.log2(nodes.shape[0]))))
kwargs = dict(n_trees=n_trees, n_iters=n_iters, n_neighbors=nneighbors, max_candidates=60, verbose=True, metric=metric)
if metric == "minkowski": kwargs['metric_kwds'] = dict(p=p)
print( f"Computing NN-Graph with parms= {kwargs}, nodes shape = {nodes.shape}, #NULL={np.count_nonzero(np.isnan(nodes.values))}" )
print( f"data ave = {nodes.values.mean()}, std = {nodes.values.std()}, range = [{nodes.values.min()},{nodes.values.max()}]")
knn_graph = NNDescent( nodes.values, **kwargs)
print(f"NN-Graph COMPLETED")
