import xarray as xa
import time, traceback, abc
from sklearn.exceptions import NotFittedError
import numpy as np
import os, copy
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from enum import Enum

class ModelType(Enum):
    SPATIAL = 1
    MODEL = 2
    SPECTRAL = 3
    SPECTRALSPATIAL = 4
    CUSTOM = 5

class LearningModel:

    def __init__(self, name: str,  mtype: ModelType, **kwargs ):
        self.mid =  name
        self.mtype = mtype
        self.device = "cpu"
        self._score: Optional[np.ndarray] = None
        self.config = kwargs
        self._keys = []
        self.classification: xa.DataArray = None
        self.confidence: xa.DataArray = None

    def clear(self):
        raise Exception( "abstract method LearningModel.clear called")

    def epoch_callback( self, epoch, logs ):
        pass

    def rebuild(self):
        self.clear()

    @property
    def model_dir(self):
        from spectraclass.data.base import dm
        from spectraclass.data.spatial.tile.manager import tm
        mdir = os.path.join( dm().cache_dir, "models", tm().tileName() )
        os.makedirs( mdir, 0o777, exist_ok=True)
        return mdir

    @property
    def model_file(self):
        mname = datetime.now().strftime(f"%Y.%j_%H.%M.%S.tf")
        return os.path.join( self.model_dir, mname )

    def list_models(self) -> Dict[str,str]:
        models = {}
        for mfile in os.listdir(self.model_dir):
            if mfile.endswith('.tf'):
                models[ os.path.splitext(mfile)[0] ] = os.path.join( self.model_dir, mfile)
        return models

    def setKeys(self, keys: List[str] ):
        self._keys = keys

    @property
    def score(self) -> Optional[np.ndarray]:
        return self._score

    def get_training_set(self, **kwargs ) -> Tuple[np.ndarray,np.ndarray,Optional[np.ndarray],Optional[np.ndarray]]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.model.labels import LabelsManager, Action, lm
        input_data: xa.DataArray = None
        label_data = lm().getTrainingLabels()
        training_data, training_labels = None, None
        for ( (tindex, block_coords, cid), gids ) in label_data.items():
            block = tm().getBlock( tindex=tindex, block_coords=block_coords )
            if   self.mtype == ModelType.MODEL:
                input_data = block.model_data
            elif self.mtype == ModelType.SPECTRAL:
                input_data = block.getPointData()[0].expand_dims("channels",2)
            elif self.mtype == ModelType.SPECTRALSPATIAL:
                input_data = block.getSpectralData(raster=True).expand_dims("samples", 0).expand_dims("channels", 4)
            else:    raise Exception( f"Unusable input data type to get_training_set: {self.mtype}")
            training_mask: np.ndarray = np.isin( input_data.samples.values, gids )
            tdata: np.ndarray = input_data.values[ training_mask ]
            tlabels: np.ndarray = np.full([gids.size], cid)
            lgm().log( f"Adding training data: tindex={tindex} bindex={block_coords} cid={cid} #gids={gids.size} data.shape={tdata.shape} labels.shape={tlabels.shape} mask.shape={training_mask.shape}")
            training_data   = tdata   if (training_data   is None) else np.append( training_data,   tdata,   axis=0 )
            training_labels = tlabels if (training_labels is None) else np.append( training_labels, tlabels, axis=0 )
        lgm().log(f"SHAPES--> training_data: {training_data.shape}, training_labels: {training_labels.shape}" )
        return ( training_data, training_labels, None, None )

    @exception_handled
    def learn_classification( self,**kwargs ):
        training_data, training_labels, sample_weight, test_mask = self.get_training_set( **kwargs )
        t1 = time.time()
        if np.count_nonzero( training_labels > 0 ) == 0:
            ufm().show( "Must label some points before learning the classification" )
            return None
        with tf.device(f'/{self.device}:0'):
            self.fit( training_data, training_labels, sample_weight=sample_weight, **kwargs )
        lgm().log(f"Completed learning in {time.time() - t1} sec.")

    @classmethod
    def index_to_one_hot( cls, class_data: np.ndarray ) -> np.ndarray:
        from spectraclass.model.labels import lm
        return to_categorical( class_data, lm().nLabels )

    @classmethod
    def one_hot_to_index(cls, class_data: np.ndarray) -> np.ndarray:
        return np.argmax( class_data, axis=0  ).squeeze()

    def fit( self, data: np.ndarray, class_data: np.ndarray, **kwargs ):
        raise Exception( "abstract method LearningModel.fit called")

    def get_input_data(self) -> xa.DataArray:
        from spectraclass.data.spatial.tile.tile import Block
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        input_data: xa.DataArray = None
        block: Block = tm().getBlock()
        if   self.mtype == ModelType.MODEL:
            input_data = block.model_data
        elif self.mtype == ModelType.SPECTRAL:
            input_data = block.getPointData()[0]
        elif self.mtype == ModelType.SPECTRALSPATIAL:
            input_data = block.getSpectralData(raster=True).expand_dims("samples", 0) ## .expand_dims("channels", 4)
        return input_data

    @exception_handled
    def apply_classification( self, **kwargs ) -> Tuple[xa.DataArray,xa.DataArray]:
        try:
            from spectraclass.gui.pointcloud import PointCloudManager, pcm
            from spectraclass.data.spatial.tile.manager import TileManager, tm
            from spectraclass.gui.spatial.map import MapManager, mm
            from spectraclass.model.labels import LabelsManager, Action, lm
            lgm().log( f" APPLY classification" )
            input_data: xa.DataArray = self.get_input_data()
            prediction, pred_confidence = self.predict( input_data.values, **kwargs )
            if pred_confidence.ndim == 1: pred_confidence = np.expand_dims(pred_confidence, 1)
            if prediction.ndim == 1: prediction = np.expand_dims(prediction, 1)
            self.classification = xa.DataArray( prediction, dims=['samples', 'classes'], attrs=input_data.attrs,
                                                coords=dict(samples=input_data.coords['samples'], classes=range(prediction.shape[1])))
            lm().addAction("classify", "application")
            self.confidence = xa.DataArray( pred_confidence, dims=['samples', 'classes'], attrs=input_data.attrs,
                                           coords=dict(samples=input_data.coords['samples'], classes=range(pred_confidence.shape[1])))
            return self.classification, self.confidence
        except NotFittedError:
            ufm().show( "Must learn a mapping before applying a classification", "warning")

    def predict( self, data: np.ndarray, **kwargs ):
        raise Exception( "abstract method LearningModel.predict called")

    def save( self, **kwargs ) -> str:
        raise Exception( "abstract method LearningModel.save called")

    @exception_handled
    def load( self, model_name: str, **kwargs ):
        raise Exception( "abstract method LearningModel.load called")


class KerasLearningModel(LearningModel):

    def __init__(self, name: str, mtype: ModelType, model: Model, callbacks: List[Callback] = None,  **kwargs ):
        LearningModel.__init__( self, name, mtype, **self.set_learning_parameters( **kwargs ) )
        self.callbacks: List[Callback] = callbacks if callbacks else []
        self.callbacks.append( lgm().get_keras_logger() )
        self.device = "cpu"
        self._model: Model = model
        self.compile()

    def set_learning_parameters( self, **kwargs ) -> Dict:
        from spectraclass.learn.models.network import Network
        self.opt = str(kwargs.pop( 'opt', 'adam' )).lower()   # 'adam' 'rmsprop'
        self.loss = str(kwargs.pop('loss', 'categorical_crossentropy')).lower()
        self.nepochs = kwargs.pop( 'nepochs', 300 )
        self.test_size = kwargs.pop( 'test_size', 0.0 )
        self.network: Network = kwargs.pop( 'network', None )
        return kwargs

    def rebuild(self):
        if self.network is not None:
            self._model, largs = self.network.build_model()
        self.compile()

    def compile(self):
        lgm().log(f"Compiling model with opt={self.opt}, loss={self.loss}")
        self._model.compile(optimizer=self.opt, loss=self.loss, metrics=['accuracy'], **self.config )

    # def train_test_split(self, data: np.ndarray, class_data: np.ndarray, test_size: float ) -> List[np.ndarray]:
    #     return train_test_split(data, class_data, test_size=test_size)

    def fit( self, data: np.ndarray, class_data: np.ndarray, **kwargs ):
        args = dict( epochs=self.nepochs, callbacks=self.callbacks, verbose=2, **self.config, **kwargs )
        if class_data.ndim == 1: class_data = self.index_to_one_hot( class_data )
        lgm().log( f"KerasLearningModel[{self.mid}].fit: data{data.shape} labels{class_data.shape} args={args}" )
        ufm().show( "Running learning algorithm... " )
        with tf.device(f'/{self.device}:0'):
            self._model.fit( data, class_data, **args )

    @exception_handled
    def load( self, model_name: str, **kwargs ):
        file_path = os.path.join( self.model_dir, f"{model_name}.tf" )
        lgm().log( f'KerasModelWrapper: loading model -> {file_path}' )
        self._model = models.load_model( file_path, **kwargs )

    def save( self, **kwargs ) -> str:
        mfile = self.model_file
        lgm().log( f'KerasModelWrapper: save weights -> {mfile}' )
        self._model.save( mfile, save_format="tf", **kwargs )
        return os.path.splitext( os.path.basename(mfile) )[0]

    def epoch_callback(self, epoch, logs):
        pass

    def predict( self, data: np.ndarray, **kwargs ) -> Tuple[np.ndarray,Optional[np.ndarray]]:
        with tf.device(f'/{self.device}:0'):
            crng = lambda x: [ np.nanmin( x ), np.nanmax( x ) ]
            predictresult = self._model.predict( data, **kwargs )
            predictresult[ np.isnan(predictresult) ] = -9999
            classresult: np.ndarray = predictresult.argmax(axis=-1)
            predictresult.sort(axis=-1)
            predictresult = predictresult.squeeze()
            maxvalue, next_maxvalue = predictresult[:, -1], predictresult[:, -2]
            confidence = maxvalue - next_maxvalue
            lgm().log( f" predict-> computed confidence: crange = {crng(confidence)}" )
            lgm().log( f"  ** maxvalue crange = {crng(maxvalue)}, next_maxvalue crange = {crng(next_maxvalue)}")
            return ( classresult, confidence )

    def apply( self, data: np.ndarray, **kwargs ) -> np.ndarray:
        waves = [ w.mean() for w in self._model.get_layer(0).get_weights() ]
        lgm().log( f"KerasLearningModel[{hex(id(self))}:{hex(id(self._model))}].apply: weights = {waves}")
        with tf.device(f'/{self.device}:0'):
            return self._model( data, **kwargs )

    def clear(self):
        self.rebuild()