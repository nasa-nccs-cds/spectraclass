from spectraclass.data.spatial.tile.tile import Block
import tensorflow as tf
# import tensorflow as tf
# keras = tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras import datasets, layers, models
from typing import List, Tuple, Optional, Dict
from spectraclass.learn.models.network import Network, ModelType

class CNN(Network):
    TYPE = ModelType.SPATIAL

    def _build_model(self, **kwargs) -> Tuple[Model,Dict]:
        from spectraclass.learn.models.spatial import SpatialModelWrapper
        nfeatures = kwargs.pop('nfeatures', 64 )
        from spectraclass.model.labels import lm
        input_shape = SpatialModelWrapper.get_input_shape()
        nclasses = lm().nLabels
        ks = kwargs.pop('kernel_size',3)
        model = models.Sequential()
        model.add( Input( shape=input_shape ) )
        model.add( layers.Conv2D( nfeatures, (ks,ks), activation='relu', padding="same" ) )
        model.add( layers.Reshape( SpatialModelWrapper.flatten(input_shape,nfeatures) ) )
        model.add( layers.Dense( nfeatures, activation='relu' ) )
        model.add( layers.Dense( nclasses, activation='softmax' ) )
        return model, kwargs


