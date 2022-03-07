import xarray as xa
import time, traceback, abc
from sklearn.model_selection import train_test_split
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
import ipywidgets as ipw
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models

class LearningModel:

    def __init__(self, name: str,  **kwargs ):
        self.mid =  name
        self._score: Optional[np.ndarray] = None
        self.config = kwargs
        self._keys = []

    @property
    def model_dir(self):
        from spectraclass.data.base import dm
        from spectraclass.data.spatial.tile.manager import tm
        mdir = os.path.join( dm().cache_dir, "models", tm().tileName() )
        os.makedirs( mdir, 0o777, exist_ok=True)
        return mdir

    @property
    def model_file(self):
        ts = datetime.now().strftime("%Y.%j_%H.%M.%S")
        return os.path.join( self.model_dir, ts )

    def list_models(self) -> List[str]:
        return os.listdir( self.model_dir )

    def setKeys(self, keys: List[str] ):
        self._keys = keys

    @property
    def score(self) -> Optional[np.ndarray]:
        return self._score

    @exception_handled
    def learn_classification( self, point_data: np.ndarray, class_data: np.ndarray, **kwargs ):
        from spectraclass.model.labels import LabelsManager, lm
        t1 = time.time()
        if np.count_nonzero( class_data > 0 ) == 0:
            ufm().show( "Must label some points before learning the classification" )
            return None
        if class_data.ndim == 1:
            class_data = self.index_to_one_hot( class_data )
        lgm().log(f"Learning mapping with shapes: point_data{point_data.shape}, class_data{class_data.shape}")
        self.fit( point_data, class_data, **kwargs )
        lgm().log(f"Completed learning in {time.time() - t1} sec.")


    @classmethod
    def index_to_one_hot( cls, class_data: np.ndarray ) -> np.ndarray:
        from spectraclass.model.labels import lm
        return to_categorical( class_data, lm().nLabels )

    @classmethod
    def one_hot_to_index(cls, class_data: np.ndarray) -> np.ndarray:
        return np.argmax( class_data, axis=0  ).squeeze()

    def fit(self, data: np.ndarray, class_data: np.ndarray, **kwargs):
        raise Exception( "abstract method LearningModel.fit called")

    @exception_handled
    def apply_classification( self, data: xa.DataArray, **kwargs ) -> xa.DataArray:
        t1 = time.time()
        prediction: np.ndarray = self.predict( data.values, **kwargs )
        if prediction.ndim == 1: prediction = prediction.reshape( [ prediction.size, 1 ] )
        lgm().log(f"Applied classication with input shape {data.shape[0]} in {time.time() - t1} sec.")
        return xa.DataArray( prediction,  dims=['samples','classes'],
                             coords=dict( samples=data.coords['samples'], classes=range(prediction.shape[1]) ) )

    def predict( self, data: np.ndarray, **kwargs ):
        raise Exception( "abstract method LearningModel.predict called")

    def save( self, **kwargs ):
        raise Exception( "abstract method LearningModel.save called")

    def load( self, name, **kwargs ):
        raise Exception( "abstract method LearningModel.load called")

class KerasModelWrapper(LearningModel):

    def __init__(self, name: str,  model: models.Model, **kwargs ):
        LearningModel.__init__( self, name,  **kwargs )
        opt = str(kwargs.pop('opt', 'adam')).lower()
        loss = str(kwargs.pop('loss', 'categorical_crossentropy')).lower()
        self._model: models.Model = model
        self._model.compile(optimizer=opt, loss=loss,  metrics=['accuracy'], **kwargs )

    def predict( self, data: np.ndarray, **kwargs ) -> np.ndarray:
        return self._model.predict( data, **kwargs )

    def save( self, **kwargs ) -> str:
        lgm().log( f'KerasModelWrapper: save weights -> {self.model_file}' )
        return self._model.save_weights( self.model_file, **kwargs )

    def load( self, model_name: str, **kwargs ):
        file_path = os.path.join( self.model_dir, model_name )
        return self._model.load_weights( file_path, **kwargs )

    def fit(self, data: np.ndarray, class_data: np.ndarray, **kwargs ):
        nepochs = kwargs.pop( 'nepochs', 25 )
        test_size = kwargs.pop( 'test_size', 0.0 )
        if test_size > 0.0:
            tx, vx, ty, vy = train_test_split( data, class_data, test_size=test_size )
            self._model.fit( tx, ty, epochs=nepochs, validation_data=(vx,vy), **kwargs )
        else:
            lgm().log( f"model.fit, shapes: point_data{data.shape}, class_data{class_data.shape} " )
            self._model.fit( data, class_data, epochs=nepochs, **kwargs )



