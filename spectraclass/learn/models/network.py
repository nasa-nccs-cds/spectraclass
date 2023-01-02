from typing import List, Tuple, Optional, Dict, Union, Type
from spectraclass.learn.base import LearningModel, ModelType
import tensorflow as tf

class Network:
    TYPE: Union[ModelType,Type[LearningModel]] = None

    def __init__(self, name: str, **kwargs ):
        self._name: str = name
        self._parms = kwargs
        self._learning_model_class: Optional[Type[LearningModel]] = None
        if isinstance( self.TYPE, ModelType ):
            self._type = self.TYPE
        else:
            self._type = ModelType.CUSTOM
            self._learning_model_class = self.TYPE
        self._learning_model: LearningModel = None

    @property
    def name(self):
        return self._name

    def build_model(self) -> Tuple[tf.keras.models.Model,Dict]:
        return self._build_model(**self._parms)

    def build( self ) -> "LearningModel":
        from .spatial import SpatialModelWrapper
        from spectraclass.learn.base import KerasLearningModel
        if self._type in [ModelType.SPATIAL, ModelType.SPECTRALSPATIAL]:
                model, bparms = self.build_model()
                self._learning_model = SpatialModelWrapper( self._name, self._type, model, callbacks=[NetworkCallback(self)], network=self, **bparms )
        if self._type in [ModelType.MODEL, ModelType.SPECTRAL]:
                model, bparms = self.build_model()
                self._learning_model = KerasLearningModel( self._name, self._type, model, callbacks=[NetworkCallback(self)], network=self, **bparms )
        if self._type == ModelType.CUSTOM:
                self._learning_model = self._learning_model_class(self._name, callbacks=[NetworkCallback(self)], **self._parms)
        return self._learning_model

    def _build_model( self, **kwargs ) -> Tuple[tf.keras.models.Model,Dict]:
        raise NotImplementedError( "Attempt to call abstract method '_build_model' on Network object")

    def epoch_callback(self, epoch, logs):
        self._learning_model.epoch_callback( epoch, logs )

    def on_train_end(self, logs):
        pass

    def on_train_begin(self, logs):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

    # for i, metric in enumerate(metrics):
    #     axs[i].plot(range(1, epoch + 2),
    #                 self.metrics[metric],
    #                 label=metric)

    def on_batch_end(self, batch, logs):
        pass

    def on_predict_end(self, logs):
        pass

class NetworkCallback(tf.keras.callbacks.Callback):

    def __init__(self, network: Network ):
        tf.keras.callbacks.Callback.__init__(self)
        self._network: Network = network

    def on_train_begin(self, logs=None):
        self._network.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self._network.on_train_end(logs)

    def on_epoch_end(self, epoch, logs=None):
        self._network.on_epoch_end(epoch,logs)
        self._network.epoch_callback(epoch,logs)

    def on_batch_end(self, batch, logs=None):
        self._network.on_batch_end(batch,logs)

    def on_predict_end(self, logs=None):
        self._network.on_predict_end(logs)