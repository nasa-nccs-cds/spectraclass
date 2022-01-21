import xarray as xa
import time, traceback, abc
from sklearn.model_selection import train_test_split
import numpy as np
import scipy, sklearn
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
import ipywidgets as ipw
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class LearningModel:

    def __init__(self, name: str,  **kwargs ):
        self.mid =  name
        self._score: Optional[np.ndarray] = None
        self.config = kwargs
        self._keys = []

    def setKeys(self, keys: List[str] ):
        self._keys = keys

    @property
    def score(self) -> Optional[np.ndarray]:
        return self._score

    @exception_handled
    def learn_classification( self, point_data: np.ndarray, class_data: np.ndarray, **kwargs ):
        t1 = time.time()
        if np.count_nonzero( class_data > 0 ) == 0:
            ufm().show( "Must label some points before learning the classification" )
            return None
        if class_data.ndim == 1:
            class_data = self.index_to_one_hot( class_data )
        self.fit( point_data, class_data, **kwargs )
        lgm().log(f"Learned mapping with {class_data.shape[0]} labels in {time.time()-t1} sec.")

    def index_to_one_hot(self, class_data: np.ndarray) -> np.ndarray:
        from spectraclass.model.labels import lm
        one_hot = np.zeros((class_data.size, lm().nLabels + 1))
        one_hot[np.arange(class_data.size), class_data] = 1
        return one_hot

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


class KerasModelWrapper(LearningModel):

    def __init__(self, name: str,  model: models.Model, **kwargs ):
        LearningModel.__init__( self, name,  **kwargs )
        opt = str(kwargs.pop('opt', 'adam')).lower()
        self._model: models.Model = model
        self._model.compile(optimizer=opt, loss=self.get_loss(**kwargs),  metrics=['accuracy'], **kwargs )

    def get_loss( self, **kwargs ):
        loss = str( kwargs.get( 'loss', 'cross_entropy' ) ).lower()
        if loss == "cross_entropy": return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else: raise Exception( f"Unknown loss: {loss}" )

    def predict( self, data: np.ndarray, **kwargs ) -> np.ndarray:
        return self._model.predict( data, **kwargs )

    def fit(self, data: np.ndarray, class_data: np.ndarray, **kwargs ):
        nepochs = kwargs.pop( 'nepochs', 25 )
        test_size = kwargs.pop( 'test_size', 0.1 )
        tx, vx, ty, vy = train_test_split( data, class_data, test_size=test_size )
        self._model.fit( tx, ty, epochs=nepochs, validation_data=(vx,vy), **kwargs )



