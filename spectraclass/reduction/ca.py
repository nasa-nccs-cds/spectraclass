import os, time, random, numpy as np
import holoviews as hv
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

import xarray as xa

TEST_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from sklearn.decomposition import PCA

class PCAReducer:

    def __init__(self, ndim: int):
        self._model: PCA = PCA( n_components=ndim )
        self.components: np.ndarray = None

    @property
    def explained_variance(self):
        return self._model.explained_variance_

    @property
    def explained_variance_ratio(self):
        return self._model.explained_variance_ratio_

    def train( self, train_input: np.ndarray ):
        self._model.fit( train_input )
        self.components: np.ndarray = self._model.components_
        self.mean = self._model.mean_
        print( f"PCAReducer.train: components shape={self.components.shape}, mean shape= {self.mean.shape}" )

    def get_reduced_features(self, train_input: np.ndarray ) -> np.ndarray:
        centered_train_input: np.ndarray = train_input - self.mean
        return np.dot( centered_train_input, self.components.T )

    def get_reproduction(self, reduced_features: np.ndarray ) -> np.ndarray:
        reproduced = np.dot( reduced_features, self.components )
        return reproduced + self.mean

    @property
    def results_dir(self):
        from spectraclass.data.base import DataManager, dm
        return dm().cache_dir

    def save(self, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        name = kwargs.get('id', tm().tileid )
        models_dir = f"{self.results_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        try:
            model_path = f"{models_dir}/{name}.pca.nc"
            print(f"Saving model {name}: {model_path}" )
            xcomponents: xa.DataArray = xa.DataArray( self.components, dims=['features','bands'] )
            xmean: xa.DataArray = xa.DataArray(self.mean, dims=['bands'])
            model_dset = xa.Dataset( dict( components=xcomponents, mean=xmean) )
            model_dset.to_netcdf( model_path )
        except Exception as err:
            print(f"Error saving model {name}: {err}")

    @exception_handled
    def get_component_graph(self) -> hv.Overlay:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.spatial.tile.tile import Block
        block: Block = tm().getBlock()
        point_data = block.filtered_point_data
        popts = dict(width=600, height=200, yaxis="bare", ylim=(-3, 3), alpha=0.6)
        graphs, colors = [], [ 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'orange' ]
        for iC in range( self.components.shape[0] ):
            component: np.ndarray = self.components[iC]
            data_table: hv.Table = hv.Table((point_data.band.values, component ), 'Band', 'PCA Component')
            comp_graph = hv.Curve(data_table).opts(line_width=2, line_color=colors[iC], **popts)
            graphs.append( comp_graph )
        result = hv.Overlay(graphs)
        return result

    def load(self, **kwargs ) -> bool:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        name = kwargs.get('id', tm().tileid )
        models_dir = f"{self.results_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        try:
            model_path = f"{models_dir}/{name}.pca.npy"
            if os.path.exists( model_path ):
                print(f"Loading model {name}: {model_path}" )
                model_dset: xa.Dataset = xa.open_dataset( model_path )
                self.components = model_dset.data_vars['components'].values
                self.mean = model_dset.data_vars['mean'].values
                return True
            else:
                return False
        except Exception as err:
            print(f"Error loading model {name}: {err}")


