import xarray as xa
import time, traceback, abc, os, copy
import numpy as np
from spectraclass.data.spatial.tile.tile import Block
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
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
        self.test_mask = None
        self.training_data = None
        self.training_labels = None
        self.sample_weight = None

    @classmethod
    def flatten( cls, shape, nfeatures ):
        if   len( shape ) == 4: return [ shape[0], shape[1]*shape[2], nfeatures ]
        elif len( shape ) == 3: return [ shape[0] * shape[1], nfeatures ]

    def get_sample_weight( self, labels: np.ndarray ) -> Tuple[np.ndarray,Optional[np.ndarray]]:
        label_mask = (labels > 0)
        sample_weights: np.ndarray = np.where(label_mask, 1.0, 0.0 )
        if self.test_size == 0.0:
            test_mask = np.full(label_mask.shape, False)
        else:
            tmask = ( np.random.rand( label_mask.size ) < self.test_size )
            test_mask = tmask & label_mask
            sample_weights[ test_mask ] = 0.0
            lgm().log( f"TMASK: tmask{tmask.shape} size={np.count_nonzero(tmask)}, test_mask{test_mask.shape} size={np.count_nonzero(test_mask)}, label_mask{label_mask.shape} size={np.count_nonzero(label_mask)}")
        return np.expand_dims(sample_weights, 0), np.expand_dims(test_mask, 0)

    # def train_test_split(self, data: np.ndarray, class_data: np.ndarray, test_size: float ) -> List[np.ndarray]:
    #     lgm().log( f"train_test_split-> data{data.shape} labels{class_data.shape}")
    #     sdata, slabels = data.squeeze(), class_data.squeeze()
    #     gshape = [ sdata[0], sdata[1] ]
    #     sdata: np.ndarray = sdata.reshape( gshape[0]*gshape[1], sdata[2] )
    #     return [ np.expand_dims(x,0) for x in train_test_split( data[0], class_data, test_size=test_size ) ]

    def get_class_weight( self, labels: np.ndarray ) -> Dict[int,float]:
        from spectraclass.model.labels import LabelsManager, lm
        nLabels = lm().nLabels
        label_counts = [np.count_nonzero(labels == iC) for iC in range(nLabels)]
        label_weights = np.array([1.0 / lc if (lc > 0.0) else 0.0 for lc in label_counts])
        weights_sum = label_weights.sum()
        label_weights = {iC: label_weights[iC] / weights_sum for iC in range(nLabels)}
        return label_weights

    def get_training_set(self, block: Block, **kwargs ) -> Tuple[np.ndarray,np.ndarray,Optional[np.ndarray],Optional[np.ndarray]]:
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.learn.base import LearningModel
        label_map: np.ndarray  = lm().get_label_map( block=block ).values.flatten()
        base_data: xa.DataArray = block.getModelData(True) if dm().use_model_data else block.getSpectralData(True)
        tdims = [ base_data.dims[1], base_data.dims[2], base_data.dims[0] ]
        lgm().log(f"get_training_set->base_data: shape={base_data.shape} dims={base_data.dims} tdims={tdims}")
        training_data: np.ndarray = base_data.transpose(*tdims).fillna(0.0).expand_dims('batch', 0).values
        training_labels = np.expand_dims( LearningModel.index_to_one_hot( label_map ), 0 )
        sample_weight, test_mask = self.get_sample_weight( label_map )
        return ( training_data, training_labels, sample_weight, test_mask )

    @classmethod
    def block_data(cls) -> xa.DataArray:
        from spectraclass.gui.spatial.map import MapManager, mm
        rdata: xa.DataArray = mm().data
        return  rdata.transpose( rdata.dims[1], rdata.dims[2], rdata.dims[0] )

    @classmethod
    def get_input_data(cls) -> xa.DataArray:
        return cls.block_data().fillna(0.0).expand_dims('batch', 0)

    @classmethod
    def get_input_shape(cls) -> Tuple[int,...]:
        from spectraclass.data.base import DataManager, dm
        input_shape =  list( cls.block_data().shape )
        if dm().use_model_data: input_shape[-1] = dm().modal.model_dims
        return tuple( input_shape )

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
        self.training_data, self.training_labels, self.sample_weight, self.test_mask = None, None, None, None
        for block in blocks:
            training_data, training_labels, sample_weight, test_mask = self.get_training_set( block, **kwargs )
            if np.count_nonzero( training_labels > 0 ) > 0:
                if self.training_data is None:
                    self.training_data, self.training_labels, self.sample_weight, self.test_mask = training_data, training_labels, sample_weight, test_mask
                else:
                    self.training_data =   np.concatenate( (self.training_data,   training_data) )
                    self.training_labels = np.concatenate( (self.training_labels, training_labels) )
                    self.sample_weight =   np.concatenate( (self.sample_weight,   sample_weight) )
                    self.test_mask =       np.concatenate( (self.test_mask,       test_mask) )
        lgm().log(f"Learning mapping with shapes: spectral_data{self.training_data.shape}, class_data{self.training_labels.shape}, sample_weight{self.sample_weight.shape}")
        self.fit( self.training_data, self.training_labels, sample_weight=self.sample_weight, **kwargs )
        lgm().log(f"Completed Spatial learning in {time.time() - t1} sec.")

    @exception_handled
    def epoch_callback(self, epoch):
        if (self.test_mask is not None) and (self.test_size > 0.0):
            for iBlock in range( self.training_data.shape[0] ):
                prediction = self.predict( self.training_data[iBlock] )
                test_results = np.equal( prediction.flatten(), self.training_labels.squeeze().argmax(axis=1) )[ self.test_mask[iBlock] ]
                accuracy = np.count_nonzero( test_results ) / test_results.size
                lgm().log( f"Epoch[{epoch}]-> BLOCK-{iBlock}: Test[{test_results.size}] accuracy: {accuracy:.4f}" )

    def predict( self, data: np.ndarray, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block: Block = tm().getBlock()
        postresult: np.ndarray = self._model.predict(data).squeeze()
        classresult: np.ndarray = postresult.argmax(axis=1)
        classresult[ ~block.raster_mask.flatten() ] = 0
        return classresult.reshape( block.raster_mask.shape )
