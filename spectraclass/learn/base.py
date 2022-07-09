import xarray as xa
import time, traceback, abc
from sklearn.model_selection import train_test_split
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
import ipywidgets as ipw
import copy
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models

class LearningModel:

    def __init__(self, name: str,  **kwargs ):
        self.mid =  name
        self._score: Optional[np.ndarray] = None
        self.config = kwargs
        self._keys = []

    def clear(self):
        raise Exception( "abstract method LearningModel.clear called")

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

    def get_training_set(self, **kwargs ) -> Tuple[np.ndarray,np.ndarray]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.model.labels import LabelsManager, Action, lm
        label_data = lm().getTrainingLabels()
        training_data, training_labels = None, None
        for ( (tindex, bindex, cid), pids ) in label_data.items():
            model_data: xa.DataArray = tm().getBlock( tindex=tindex, bindex=bindex ).model_data
            training_mask: np.ndarray = np.isin( model_data.samples.values, pids )
            tdata: np.ndarray = model_data.values[ training_mask ]
            lgm().log( f"Adding training data: tindex={tindex},  bindex={bindex},  cid={cid},  #pids={pids.size} ")
            tlabels: np.ndarray = np.full( [pids.size], cid )
            training_data   = tdata   if (training_data   is None) else np.append( training_data,   tdata,   axis=0 )
            training_labels = tlabels if (training_labels is None) else np.append( training_labels, tlabels, axis=0 )
        lgm().log(f"SHAPES--> training_data: {training_data.shape}, training_labels: {training_labels.shape}" )
        return ( training_data, training_labels )

    @exception_handled
    def learn_classification( self,**kwargs ):
        training_data, training_labels = self.get_training_set( **kwargs )
        t1 = time.time()
        if np.count_nonzero( training_labels > 0 ) == 0:
            ufm().show( "Must label some points before learning the classification" )
            return None
        lgm().log(f"Learning mapping with shapes: spectral_data{training_data.shape}, class_data{training_labels.shape}")
        self.fit( training_data, training_labels, **kwargs )
        lgm().log(f"Completed learning in {time.time() - t1} sec.")


    @classmethod
    def index_to_one_hot( cls, class_data: np.ndarray ) -> np.ndarray:
        from spectraclass.model.labels import lm
        return to_categorical( class_data, lm().nLabels )

    @classmethod
    def one_hot_to_index(cls, class_data: np.ndarray) -> np.ndarray:
        return np.argmax( class_data, axis=0  ).squeeze()

    def fit(self, data: np.ndarray, class_data: np.ndarray, **kwargs):
        raise Exception( "abstract method LearningModel.fit called")

    @exception_handled
    def apply_classification( self, data: xa.DataArray, **kwargs ) -> xa.DataArray:
        t1 = time.time()
        prediction: np.ndarray = self.predict( data.values, **kwargs )
        if prediction.ndim == 1: prediction = prediction.reshape( [ prediction.size, 1 ] )
        lgm().log(f"Applied classication with input shape {data.shape[0]} in {time.time() - t1} sec.")
        return xa.DataArray( prediction,  dims=['samples','classes'],
                             coords=dict( samples=data.coords['samples'], classes=range(prediction.shape[1]) ) )

    def predict( self, data: np.ndarray, **kwargs ):
        raise Exception( "abstract method LearningModel.predict called")

    def save( self, **kwargs ):
        raise Exception( "abstract method LearningModel.save called")

    def load( self, name, **kwargs ):
        raise Exception( "abstract method LearningModel.load called")

class KerasModelWrapper(LearningModel):

    def __init__(self, name: str,  model: models.Model, **kwargs ):
        LearningModel.__init__( self, name,  **kwargs )
        self.opt = str(kwargs.pop('opt', 'adam')).lower()
        self.loss = str(kwargs.pop('loss', 'categorical_crossentropy')).lower()
        self.parms = kwargs
        self._model: models.Model = model
        self._model.compile(optimizer=self.opt, loss=self.loss,  metrics=['accuracy'], **kwargs )
        self._init_model = copy.deepcopy(model)

    def predict( self, data: np.ndarray, **kwargs ) -> np.ndarray:
        return self._model.predict( data, **kwargs )

    def save( self, **kwargs ) -> str:
        mfile = self.model_file
        lgm().log( f'KerasModelWrapper: save weights -> {mfile}' )
        self._model.save( mfile, save_format="tf", **kwargs )
        return os.path.splitext( os.path.basename(mfile) )[0]

    @exception_handled
    def load( self, model_name: str, **kwargs ):
        file_path = os.path.join( self.model_dir, f"{model_name}.tf" )
        lgm().log( f'KerasModelWrapper: loading model -> {file_path}' )
        self._model = models.load_model( file_path, **kwargs )

    def clear(self):
        self._model = self._init_model
        self._model.compile(optimizer=self.opt, loss=self.loss,  metrics=['accuracy'], **self.parms )

    def fit(self, data: np.ndarray, class_data: np.ndarray, **kwargs ):
        nepochs = kwargs.pop( 'nepochs', 25 )
        test_size = kwargs.pop( 'test_size', 0.0 )
        if class_data.ndim == 1:
            class_data = self.index_to_one_hot( class_data )
        if test_size > 0.0:
            tx, vx, ty, vy = train_test_split( data, class_data, test_size=test_size )
            self._model.fit( tx, ty, epochs=nepochs, validation_data=(vx,vy), **kwargs )
        else:
            lgm().log( f"model.fit, shapes: point_data{data.shape}, class_data{class_data.shape} " )
            self._model.fit( data, class_data, epochs=nepochs, **kwargs )



