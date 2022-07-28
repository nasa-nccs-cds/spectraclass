import xarray as xa
import time, traceback, abc
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import numpy as np
from typing import List, Tuple, Optional, Dict
import copy
from spectraclass.learn.base import LearningModel
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras import datasets, layers, models

class SamplesModelWrapper(LearningModel):

    def __init__( self, name: str,  model: models.Model, **kwargs ):
        LearningModel.__init__( self, name,  **kwargs )
        self.opt = str(kwargs.pop('opt', 'adam')).lower()
        self.loss = str(kwargs.pop('loss', 'categorical_crossentropy')).lower()
        self.spatial = kwargs.get( 'spatial', False )
        self.parms = kwargs
        self._model: models.Model = model
        self._model.compile( optimizer=self.opt, loss=self.loss,  metrics=['accuracy'], **kwargs )
        self._init_model = copy.deepcopy(model)

    def predict( self, data: np.ndarray, **kwargs ) -> np.ndarray:
        return self._model.predict( data, **kwargs )

    def apply( self, data: np.ndarray, **kwargs ) -> np.ndarray:
        return self._model( data, **kwargs )

    def clear(self):
        self._model = self._init_model
        self._model.compile(optimizer=self.opt, loss=self.loss,  metrics=['accuracy'], **self.parms )

    def fit(self, data: np.ndarray, class_data: np.ndarray, **kwargs ):
        nepochs = kwargs.pop( 'nepochs', 25 )
        test_size = kwargs.pop( 'test_size', 0.0 )
        if class_data.ndim == 1:
            class_data = self.index_to_one_hot( class_data )
        if test_size > 0.0:
            tx, vx, ty, vy = train_test_split( data, class_data, test_size=test_size )
            self._model.fit( tx, ty, epochs=nepochs, validation_data=(vx,vy), **kwargs )
        else:
            lgm().log( f"model.fit, shapes: point_data{data.shape}, class_data{class_data.shape} " )
            self._model.fit( data, class_data, epochs=nepochs, **kwargs )

