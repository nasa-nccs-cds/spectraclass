import xarray as xa
import time, traceback, abc, os, copy
import numpy as np
from spectraclass.data.spatial.tile.tile import Block
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from typing import List, Tuple, Optional, Dict
import traitlets as tl
import traitlets.config as tlc
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.learn.base import LearningModel
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras import datasets, layers, models
from .base import LearningModel, SamplesModelWrapper

class CNN:

    @classmethod
    def build( cls, nfeatures: int, **kwargs ) -> models.Sequential:    # (batch, channels, height, width)
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.model.labels import lm
        block: Block = tm().getBlock()
        input_shape = block.data.transpose('y', 'x', 'band').shape
        nclasses = lm().nLabels
        lgm().log( f"CNN.build: input_shape={input_shape}, nfeatures={nfeatures}, nclasses={nclasses}" )
        ks = kwargs.get('kernel_size',3)
        model = models.Sequential()
        model.add( Input( shape=input_shape ) )
        model.add( layers.Conv2D( nfeatures, (ks,ks), activation='tanh', padding="same" ) )
        model.add( layers.Reshape( cls.flatten(input_shape,nfeatures) ) )
        model.add( layers.Dense( nfeatures, activation='tanh' ) )
#        model.add( layers.Dense( nfeatures//2, activation='tanh') )
        model.add( layers.Dense( nclasses, activation='softmax' ) )
        return model

    @classmethod
    def flatten( cls, shape, nfeatures ):
        if   len( shape ) == 4: return [ shape[0], shape[1]*shape[2], nfeatures ]
        elif len( shape ) == 3: return [ shape[0] * shape[1], nfeatures ]

    @classmethod
    def get_block_data(cls) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block: Block = tm().getBlock()
        bdata: xa.DataArray = block.data.transpose('y', 'x', 'band').fillna(0.0)
        return bdata.expand_dims('batch',0)

class SpatialModelWrapper(LearningModel):

    def __init__(self, name: str,  model: models.Model, **kwargs ):
        LearningModel.__init__( self, name,  **kwargs )
        self.opt = str(kwargs.pop('opt', 'adam')).lower()
        self.loss = str(kwargs.pop('loss', 'categorical_crossentropy')).lower()
        self.parms = kwargs
        self._model: models.Model = model
        self._model.compile( self.opt, self.loss,  metrics=['accuracy'], **kwargs )
        self._init_model = copy.deepcopy(model)

    def get_sample_weights( self, labels: np.ndarray, nLabels: int) -> np.ndarray:
        sample_weights: np.ndarray = np.where((labels == 0), 0.0, 1.0)
        label_counts = [np.count_nonzero(labels == iC) for iC in range(nLabels)]
        label_weights = np.array([1.0 / lc if (lc > 0.0) else 0.0 for lc in label_counts])
        weights_sum = label_weights.sum()
        label_weights = {iC: label_weights[iC] / weights_sum for iC in range(nLabels)}
        for iC in range(1, nLabels):
            if label_weights[iC] > 0.0:
                sample_weights[(labels == iC)] = label_weights[iC]
        return np.expand_dims(sample_weights, 0)

    def get_training_set(self, block: Block, **kwargs ) -> Tuple[np.ndarray,np.ndarray]:
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.learn.base import LearningModel
        label_map: xa.DataArray  = lm().get_label_map( block=block )
        training_data: np.ndarray = block.data.transpose('y', 'x', 'band').fillna(0.0).expand_dims('batch', 0).values
        training_labels = np.expand_dims( LearningModel.index_to_one_hot( label_map.values.flatten() ), 0 )
        return ( training_data, training_labels )

    def get_input_data(self) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block: Block = tm().getBlock()
        input_data: xa.DataArray = block.data.transpose('y', 'x', 'band').fillna(0.0).expand_dims('batch', 0)
        return input_data

    @exception_handled
    def apply_classification( self, **kwargs ) -> xa.DataArray:
        try:
            from spectraclass.gui.pointcloud import PointCloudManager, pcm
            from spectraclass.data.spatial.tile.manager import TileManager, tm
            from spectraclass.gui.spatial.map import MapManager, mm
            from spectraclass.model.labels import LabelsManager, Action, lm
            input_data: xa.DataArray = self.get_input_data()
            prediction: np.ndarray = self.predict( input_data.values, **kwargs )
            classifcation: np.ndarray = prediction.reshape( [input_data.shape[1], input_data.shape[2], prediction.shape[-1]] )
            lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> CLASSIFY, result shape = {classifcation.shape}, vrange = [{prediction.min()}, {prediction.max()}] ")
            classification = xa.DataArray(  classifcation.argmax( axis=2 ).squeeze(),
                                            dims=['y', 'x' ],
                                            coords=dict( y= input_data.coords['y'],
                                                         x= input_data.coords['x'] ) )
            mm().plot_labels_image( classification )
            lm().addAction("classify", "application")
            return classification
        except NotFittedError:
            ufm().show( "Must learn a mapping before applying a classification", "red")

    @exception_handled
    def learn_classification( self,**kwargs ):
        from spectraclass.data.spatial.tile.tile import Block
        from spectraclass.model.labels import lm
        t1 = time.time()
        blocks: List[Block] = lm().getTrainingBlocks()
        for block in blocks:
            training_data, training_labels = self.get_training_set( block, **kwargs )
            if np.count_nonzero( training_labels > 0 ) > 0:
                lgm().log(f"Learning mapping with shapes: spectral_data{training_data.shape}, class_data{training_labels.shape}")
                self.fit( training_data, training_labels, **kwargs )
        lgm().log(f"Completed Spatial learning in {time.time() - t1} sec.")


    def fit( self, data: np.ndarray, class_data: np.ndarray, **kwargs ):
        from spectraclass.model.labels import LabelsManager, lm
        nepochs = kwargs.pop( 'nepochs', 25 )
        test_size = kwargs.pop( 'test_size', 0.0 )
        sample_weights: np.ndarray = self.get_sample_weights( class_data, lm().nLabels )
        self._model.fit( data, class_data, sample_weight=sample_weights, epochs=nepochs )

    def predict( self, data: np.ndarray, **kwargs ):
        postresult: np.ndarray = self._model.predict(data).squeeze()
        classresult: np.ndarray = postresult.argmax(axis=1)    # .reshape(labels_array.shape)
        return classresult
