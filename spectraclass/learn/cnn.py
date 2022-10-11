from spectraclass.data.spatial.tile.tile import Block
import tensorflow as tf
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from typing import List, Tuple, Optional, Dict
from spectraclass.learn.models.network import Network, ModelType

class CNN(Network):
    TYPE = ModelType.SPATIAL

    def _build_model(self, **kwargs) -> Tuple[tf.keras.models.Model,Dict]:
        from spectraclass.learn.models.spatial import SpatialModelWrapper
        nfeatures = kwargs.pop('nfeatures', 64 )
        from spectraclass.model.labels import lm
        input_shape = SpatialModelWrapper.get_input_shape()
        nclasses = lm().nLabels
        ks = kwargs.pop('kernel_size',3)
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.Input( shape=input_shape ) )
        model.add( tf.keras.layers.Conv2D( nfeatures, (ks,ks), activation='relu', padding="same" ) )
        model.add( tf.keras.layers.Reshape( SpatialModelWrapper.flatten(input_shape,nfeatures) ) )
        model.add( tf.keras.layers.Dense( nfeatures, activation='relu' ) )
        model.add( tf.keras.layers.Dense( nclasses, activation='softmax' ) )
        return model, kwargs


