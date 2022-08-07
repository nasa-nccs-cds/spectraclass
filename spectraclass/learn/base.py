import xarray as xa
import time, traceback, abc
from sklearn.exceptions import NotFittedError
import numpy as np
import os, copy
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import Callback

class LearningModel:

    def __init__(self, name: str,  **kwargs ):
        self.mid =  name
        self._score: Optional[np.ndarray] = None
        self.config = kwargs
        self._keys = []

    def clear(self):
        raise Exception( "abstract method LearningModel.clear called")

    def epoch_callback(self, epoch):
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
        label_data = lm().getTrainingLabels()
        training_data, training_labels = None, None
        for ( (tindex, bindex, cid), pids ) in label_data.items():
            model_data: xa.DataArray = tm().getBlock( tindex=tindex, bindex=bindex ).model_data
            training_mask: np.ndarray = np.isin( model_data.samples.values, pids )
            tdata: np.ndarray = model_data.values[ training_mask ]
            tlabels: np.ndarray = np.full([pids.size], cid)
            lgm().log( f"Adding training data: tindex={tindex} bindex={bindex} cid={cid} #pids={pids.size} data.shape={tdata.shape} labels.shape={tlabels.shape} mask.shape={training_mask.shape}")
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
        lgm().log(f"Learning mapping with shapes: spectral_data{training_data.shape}, class_data{training_labels.shape}")
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
        from spectraclass.data.base import DataManager, dm
        embedding: xa.DataArray = dm().getModelData()
        return embedding

    @exception_handled
    def apply_classification( self, **kwargs ) -> xa.DataArray:
        try:
            from spectraclass.gui.pointcloud import PointCloudManager, pcm
            from spectraclass.data.spatial.tile.manager import TileManager, tm
            from spectraclass.gui.spatial.map import MapManager, mm
            from spectraclass.model.labels import LabelsManager, Action, lm
            lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> CLASSIFY ")
            block = tm().getBlock()
            input_data: xa.DataArray = self.get_input_data()
            prediction: np.ndarray = self.predict( input_data.values, **kwargs )
            classification = xa.DataArray(prediction, dims=['samples', 'classes'], coords=dict(samples=input_data.coords['samples'], classes=range(prediction.shape[1])))
            if classification.ndim == 1: classification = classification.reshape([classification.size, 1])
            overlay_image: xa.DataArray = block.points2raster(classification)
            mm().plot_labels_image(overlay_image)
            lm().addAction("classify", "application")
     #       lm().set_classification( np.argmax(prediction, axis=1) )
            return classification
        except NotFittedError:
            ufm().show( "Must learn a mapping before applying a classification", "red")

    def predict( self, data: np.ndarray, **kwargs ):
        raise Exception( "abstract method LearningModel.predict called")

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


class KerasLearningModel(LearningModel):

    def __init__(self, name: str, model: Model, callbacks: List[Callback] = None,  **kwargs ):
        LearningModel.__init__( self, name, **self.set_learning_parameters( **kwargs ) )
        self.callbacks: List[Callback] = callbacks if callbacks else []
        self.callbacks.append( lgm().get_keras_logger() )
        self._init_model = model
        self.compile()

    def set_learning_parameters( self, **kwargs ) -> Dict:
        from spectraclass.learn.models.network import Network
        self.opt = str(kwargs.pop('opt', 'adam')).lower()
        self.loss = str(kwargs.pop('loss', 'categorical_crossentropy')).lower()
        self.nepochs = kwargs.pop( 'nepochs', 32 )
        self.test_size = kwargs.pop( 'test_size', 0.0 )
        self.network: Network = kwargs.pop( 'network', None )
        return kwargs

    def rebuild(self):
        if self.network is not None:
            self._init_model, largs = self.network.build_model()
        self.compile()

    def compile(self):
        self._model = copy.deepcopy( self._init_model )
        self._model.compile(optimizer=self.opt, loss=self.loss, metrics=['accuracy'], **self.config )

    # def train_test_split(self, data: np.ndarray, class_data: np.ndarray, test_size: float ) -> List[np.ndarray]:
    #     return train_test_split(data, class_data, test_size=test_size)

    def fit( self, data: np.ndarray, class_data: np.ndarray, **kwargs ):
        args = dict( epochs=self.nepochs, callbacks=self.callbacks, verbose=2, **self.config, **kwargs )
        if class_data.ndim == 1: class_data = self.index_to_one_hot( class_data )
        lgm().log( f"KerasLearningModel.fit: data{data.shape} labels{class_data.shape} args={args}" )
        self._model.fit( data, class_data, **args )

    def epoch_callback(self, epoch):
        pass

    def predict( self, data: np.ndarray, **kwargs ) -> np.ndarray:
        return self._model.predict( data, **kwargs )

    def apply( self, data: np.ndarray, **kwargs ) -> np.ndarray:
        return self._model( data, **kwargs )

    def clear(self):
        self.compile()