import xarray as xa
import time, traceback, abc
import numpy as np
import scipy, sklearn
# import tensorflow as tf
# keras = tf.keras
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
import traitlets as tl
import traitlets.config as tlc
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras import datasets, layers, models
from spectraclass.learn.models.network import Network
from spectraclass.learn.base import ModelType

class MLP(Network):
    TYPE = ModelType.MODEL

    def _build_model( self, **kwargs ) -> Tuple[Model,Dict]:
        from spectraclass.model.labels import lm
        nfeatures = kwargs.pop('nfeatures', 32)
        nclasses = lm().nLabels
        lgm().log( f"MLP.build: nfeatures={nfeatures}, nclasses={nclasses}" )
        model = models.Sequential()
        model.add( layers.Dense( nfeatures, activation='relu', input_shape=(nfeatures,) ) )
        model.add( layers.Dense( nfeatures, activation='relu') )
        model.add( layers.Dense( nclasses, activation='softmax' ) )
        return model, kwargs