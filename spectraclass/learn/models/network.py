from enum import Enum
from typing import List, Tuple, Optional, Dict, Union, Type
from spectraclass.learn.base import LearningModel
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

class ModelType(Enum):
     SPATIAL = 1
     SAMPLES = 2
     CUSTOM = 3

class Network:
    TYPE: Union[ModelType,Type[LearningModel]] = None

    def __init__(self, name: str, **kwargs ):
        self._name: str = name
        self._parms = kwargs
        self._learning_model: Optional[Type[LearningModel]] = None
        if isinstance( self.TYPE, ModelType ):
            self._type = self.TYPE
        else:
            self._type = ModelType.CUSTOM
            self._learning_model = self.TYPE

    @property
    def name(self):
        return self._name

    def build_model(self) -> Tuple[Model,Dict]:
        return self._build_model(**self._parms)

    def build( self ) -> "LearningModel":
        from .spatial import SpatialModelWrapper
        from spectraclass.learn.base import KerasLearningModel
        if self._type == ModelType.SPATIAL:
                model, bparms = self.build_model()
                return SpatialModelWrapper( self._name, model, callbacks=[NetworkCallback(self)], network=self, **bparms )
        if self._type == ModelType.SAMPLES:
                model, bparms = self.build_model()
                return KerasLearningModel( self._name, model, callbacks=[NetworkCallback(self)], network=self, **bparms )
        if self._type == ModelType.CUSTOM:
                return self._learning_model( self._name, callbacks=[NetworkCallback(self)], **self._parms )

    def _build_model( self, **kwargs ) -> Tuple[Model,Dict]:
        raise NotImplementedError( "Attempt to call abstract method '_build_model' on Network object")

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch ):
        pass

    def on_batch_end(self, batch):
        pass

    def on_predict_end(self):
        pass

class NetworkCallback(Callback):

    def __init__(self, network: Network ):
        Callback.__init__(self)
        self._network: Network = network

    def on_train_end(self, logs=None):
        self._network.on_train_end()

    def on_epoch_end(self, epoch, logs=None):
        self._network.on_epoch_end(epoch)

    def on_batch_end(self, batch, logs=None):
        self._network.on_batch_end(batch)

    def on_predict_end(self, logs=None):
        self._network.on_predict_end()