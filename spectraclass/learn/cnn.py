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

class CNN:

    @classmethod
    def build( cls, input_shape: List[int], nclasses: int, **kwargs ) -> models.Sequential:    # (batch, channels, height, width)
        ks = kwargs.get('kernel_size',3)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (ks,ks), activation='relu', input_shape=input_shape, data_format="channels_first"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten(data_format="channels_first") )
        model.add(layers.Dense(64, activation='relu'))
        model.add( layers.Dense( nclasses, activation='linear' ) )
        activation = tf.keras.layers.Softmax( axis=-1, **kwargs )
        model.add( layers.Dense( nclasses, activation=activation ))
        return model