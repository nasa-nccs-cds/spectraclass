import xarray as xa
import time, traceback, abc
import numpy as np
import scipy, sklearn
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
import traitlets as tl
import traitlets.config as tlc
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.model.labels import lm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras import datasets, layers, models

class MLP(models.Model):

    @classmethod
    def build( cls, input_shape: List[int], **kwargs ) -> models.Sequential:    # (batch, channels, height, width)
        lgm().log( f"MLP.build: {input_shape}" )
        model = models.Sequential()
        model.add( layers.Dense(input_shape[1], activation='relu', input_shape=input_shape ) )
        model.add( layers.Dense(input_shape[1], activation='relu') )
        model.add( layers.Dense( lm().nLabels ) )
        return model