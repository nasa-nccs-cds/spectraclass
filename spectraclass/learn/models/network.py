from enum import Enum
from typing import List, Tuple, Optional, Dict, Union, Type
from spectraclass.learn.base import LearningModel
from tensorflow.keras.models import Model

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

    def build( self, **kwargs ) -> "LearningModel":
        from .spatial import SpatialModelWrapper
        from .samples import SamplesModelWrapper
        match self._type:
            case ModelType.SPATIAL:
                model, bparms = self._build_model( **self._parms, **kwargs )
                return SpatialModelWrapper( self._name, model, **bparms )
            case ModelType.SAMPLES:
                model, bparms = self._build_model( **self._parms, **kwargs )
                return SamplesModelWrapper( self._name, model, **bparms )
            case ModelType.CUSTOM:
                return self._learning_model( self._name, **self._parms, **kwargs )

    def _build_model( self, **kwargs ) -> Tuple[Model,Dict]:
        raise NotImplementedError( "Attempt to call abstract method '_build_model' on Network object")
