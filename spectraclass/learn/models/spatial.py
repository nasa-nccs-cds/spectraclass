import xarray as xa
import time, traceback, abc, os, copy
import numpy as np
from spectraclass.data.spatial.tile.tile import Block
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from typing import List, Tuple, Optional, Dict
import traitlets as tl
import traitlets.config as tlc
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.learn.base import KerasLearningModel
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from tensorflow.keras import datasets, layers, models

class SpatialModelWrapper(KerasLearningModel):

    def __init__(self, name: str,  model: models.Model, **kwargs ):
        KerasLearningModel.__init__( self, name,  model, kwargs.pop('callbacks'), **kwargs )



    @classmethod
    def flatten( cls, shape, nfeatures ):
        if   len( shape ) == 4: return [ shape[0], shape[1]*shape[2], nfeatures ]
        elif len( shape ) == 3: return [ shape[0] * shape[1], nfeatures ]

    def get_sample_weight( self, labels: np.ndarray ) -> np.ndarray:
        sample_weights: np.ndarray = np.where((labels == 0), 0.0, 1.0)
        return np.expand_dims(sample_weights, 0)

    def get_class_weight( self, labels: np.ndarray ) -> Dict[int,float]:
        from spectraclass.model.labels import LabelsManager, lm
        nLabels = lm().nLabels
        label_counts = [np.count_nonzero(labels == iC) for iC in range(nLabels)]
        label_weights = np.array([1.0 / lc if (lc > 0.0) else 0.0 for lc in label_counts])
        weights_sum = label_weights.sum()
        label_weights = {iC: label_weights[iC] / weights_sum for iC in range(nLabels)}
        return label_weights

    def get_training_set(self, block: Block, **kwargs ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.learn.base import LearningModel
        label_map: np.ndarray  = lm().get_label_map( block=block ).values.flatten()
        training_data: np.ndarray = block.data.transpose('y', 'x', 'band').fillna(0.0).expand_dims('batch', 0).values
        training_labels = np.expand_dims( LearningModel.index_to_one_hot( label_map ), 0 )
        sample_weight = self.get_sample_weight( label_map )
        return ( training_data, training_labels, sample_weight )

    def get_input_data(self) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block: Block = tm().getBlock()
        input_data: xa.DataArray = block.data.transpose('y', 'x', 'band').fillna(0.0).expand_dims('batch', 0)
        return input_data

    @classmethod
    def get_input_shape(cls) -> Tuple[int,...]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block: Block = tm().getBlock()
        return block.data.transpose( 'y', 'x', 'band' ).shape

    @exception_handled
    def apply_classification( self, **kwargs ) -> xa.DataArray:
        try:
            from spectraclass.gui.pointcloud import PointCloudManager, pcm
            from spectraclass.data.spatial.tile.manager import TileManager, tm
            from spectraclass.gui.spatial.map import MapManager, mm
            from spectraclass.model.labels import LabelsManager, Action, lm
            input_data: xa.DataArray = self.get_input_data()
            classifcation: np.ndarray = self.predict( input_data.values, **kwargs )
            lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> CLASSIFY, result shape = {classifcation.shape}, vrange = [{classifcation.min()}, {classifcation.max()}] ")
            classification = xa.DataArray(  classifcation,
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
            training_data, training_labels, sample_weight = self.get_training_set( block, **kwargs )
            if np.count_nonzero( training_labels > 0 ) > 0:
                lgm().log(f"Learning mapping with shapes: spectral_data{training_data.shape}, class_data{training_labels.shape}, sample_weight{sample_weight.shape}")
                self.fit( training_data, training_labels, sample_weight=sample_weight, **kwargs )
        lgm().log(f"Completed Spatial learning in {time.time() - t1} sec.")

    def predict( self, data: np.ndarray, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block: Block = tm().getBlock()
        postresult: np.ndarray = self._model.predict(data).squeeze()
        classresult: np.ndarray = postresult.argmax(axis=1)
        classresult[ ~block.raster_mask.flatten() ] = 0
        return classresult.reshape( block.raster_mask.shape )