import xarray as xa
import time, traceback, abc
import numpy as np
import scipy, sklearn
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from typing import List, Tuple, Optional, Dict
import traitlets as tl
import traitlets.config as tlc
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.learn.base import LearningModel
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras import datasets, layers, models

class CNN:

    @classmethod
    def build( cls, input_shape, nfeatures: int, nclasses: int, **kwargs ) -> models.Sequential:    # (batch, channels, height, width)
        lgm().log( f"CNN.build: input_shape={input_shape}, nfeatures={nfeatures}, nclasses={nclasses}" )
        ks = kwargs.get('kernel_size',3)
        model = models.Sequential()
        model.add( Input( shape=input_shape ) )
        model.add( layers.Conv2D( nfeatures, (ks,ks), activation='tanh', padding="same" ) )
        model.add( layers.Reshape( cls.flatten(input_shape,nfeatures) ) )
        model.add( layers.Dense( nfeatures, activation='tanh' ) )
#        model.add( layers.Dense( nfeatures//2, activation='tanh') )
        model.add( layers.Dense( nclasses, activation='softmax' ) )
        return model

    @classmethod
    def flatten( cls, shape, nfeatures ):
        if   len( shape ) == 4: return [ shape[0], shape[1]*shape[2], nfeatures ]
        elif len( shape ) == 3: return [ shape[0] * shape[1], nfeatures ]

class CNNLearningModel(LearningModel):

    def __init__(self, **kwargs):
        LearningModel.__init__(self, "cnn", **kwargs)
        self._score: Optional[np.ndarray] = None
        self._cnn = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):  # X[n_samples, n_features], y[n_samples]
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.data.base import DataManager, dm
        nfeatures = kwargs.get('nfeatures', 32)
        if self._cnn is None:
            self._cnn = CNN.build( X.shape, nfeatures, lm().nLabels )



