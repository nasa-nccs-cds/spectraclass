from spectraclass.data.spatial.tile.tile import Block
import tensorflow as tf
# import tensorflow as tf
# keras = tf.keras
from tensorflow.keras.models import Model
from keras import Input
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras import datasets, layers, models
from typing import List, Tuple, Optional, Dict
from spectraclass.learn.models.network import Network, ModelType

class CNN(Network):
    TYPE = ModelType.SPATIAL

    def _build_model(self, **kwargs) -> Tuple[Model,Dict]:
        from spectraclass.learn.models.spatial import SpatialModelWrapper
        nfeatures = kwargs.pop('nfeatures', 32 )
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.model.labels import lm
        block: Block = tm().getBlock()
        input_shape = block.data.transpose('y', 'x', 'band').shape
        nclasses = lm().nLabels
        lgm().log( f"CNN.build: input_shape={input_shape}, nfeatures={nfeatures}, nclasses={nclasses}" )
        ks = kwargs.pop('kernel_size',3)
        model = models.Sequential()
        model.add( Input( shape=input_shape ) )
        model.add( layers.Conv2D( nfeatures, (ks,ks), activation='tanh', padding="same" ) )
        model.add( layers.Reshape( SpatialModelWrapper.flatten(input_shape,nfeatures) ) )
        model.add( layers.Dense( nfeatures, activation='tanh' ) )
#        model.add( layers.Dense( nfeatures//2, activation='tanh') )
        model.add( layers.Dense( nclasses, activation='softmax' ) )
        return model, kwargs



