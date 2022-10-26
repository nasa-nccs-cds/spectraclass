import xarray as xa
import time, traceback, abc, os, copy
import numpy as np
from spectraclass.data.spatial.tile.tile import Block
from sklearn.exceptions import NotFittedError
import tensorflow as tf
# import tensorflow as tf
# keras = tf.keras
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
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
        self._training_layers: Dict[Tuple,int] = {}

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

    @exception_handled
    def get_training_set(self, **kwargs ) -> Tuple[Optional[np.ndarray],Optional[np.ndarray],Optional[np.ndarray],Optional[np.ndarray]]:
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.learn.base import LearningModel
        training_data, training_labels, sample_weight, test_mask = None, None, None, None
        label_blocks: List[Block] = lm().getTrainingBlocks()
        if len(label_blocks) == 0:
            ufm().show( "Must label some points for learning","red")
        else:
            lgm().log(f">>> get_training_set: blocks={[b.index for b in label_blocks]}")
            for block in label_blocks:
                label_map: np.ndarray  = lm().get_label_map( block=block ).values.flatten()
                base_data: xa.DataArray = block.getModelData(True) if dm().use_model_data else block.getSpectralData(True)
                tdims = [ base_data.dims[1], base_data.dims[2], base_data.dims[0] ]
                tdata: np.ndarray = base_data.transpose(*tdims).fillna(0.0).expand_dims('batch', 0).values
                tlabels = np.expand_dims( LearningModel.index_to_one_hot( label_map ), 0 )
                weights, mask = self.get_sample_weight( label_map )
                lgm().log(f"    ->>> {(block.tile_index,block.block_coords)}->base_data: shape={base_data.shape} "
                          f"dims={base_data.dims} tdims={tdims}, nlabels={np.count_nonzero(training_labels)}, "
                          f"data_shape: {tdata.shape}, label_shape: {tlabels.shape}, weights_shape: {weights.shape},  mask_shape: {mask.shape}")
                training_data   = tdata   if (training_data   is None) else np.append( training_data,   tdata,   axis=0 )
                training_labels = tlabels if (training_labels is None) else np.append( training_labels, tlabels, axis=0 )
                sample_weight   = weights if (sample_weight   is None) else np.append( sample_weight, weights, axis=0 )
                test_mask       = mask    if (test_mask       is None) else np.append( test_mask, mask, axis=0 )
                self.set_training_layer_index( block, training_data.shape[0] - 1 )
            lgm().log( f">>> MERGED datashape: {training_data.shape}, label_shape: {training_labels.shape}, "
                       f"weights_shape: {sample_weight.shape},  mask_shape: {test_mask.shape}" )
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
            block: Block = tm().getBlock()
            classifcation: np.ndarray = self.predict( input_data.values, log=True, **kwargs )
            lgm().log( f" APPLY classification: block={block.block_coords}, result shape = {classifcation.shape}, vrange = [{classifcation.min()}, {classifcation.max()}] " )
            self.classification = xa.DataArray(  classifcation, dims=[ 'blocks', 'y', 'x' ],
                                            coords=dict( blocks = range(classifcation.shape[0]), y= input_data.coords['y'], x= input_data.coords['x'] ) )
#            block_index = self.get_training_layer_index( block )
            mm().plot_labels_image( self.classification[0] )
            lm().addAction("classify", "application")
            return self.classification
        except NotFittedError:
            ufm().show( "Must learn a mapping before applying a classification", "red")

    @exception_handled
    def learn_classification( self,**kwargs ):
        def count_nan(array: np.ndarray): return np.count_nonzero( np.isnan(array) )
        t1 = time.time()
        self.training_data, self.training_labels, self.sample_weight, self.test_mask = self.get_training_set( **kwargs )
        if self.training_data is not None:
            lgm().log(f"Learning mapping with shapes: spectral_data{self.training_data.shape}, class_data{self.training_labels.shape}, sample_weight{self.sample_weight.shape}")
            lgm().log( f"#NaN: spectral_data={count_nan(self.training_data)}, class_data={count_nan(self.training_labels)}, sample_weight={count_nan(self.sample_weight)}")
            self.fit( self.training_data, self.training_labels, sample_weight=self.sample_weight, **kwargs )
            lgm().log(f"Completed Spatial learning in {time.time() - t1} sec.")

    @classmethod
    def concat( cls, base_array: np.ndarray, new_array: np.ndarray) -> np.ndarray:
        return new_array if (base_array is None) else np.concatenate( (base_array, new_array) )

    def set_training_layer_index(self, block: Block, layer_index ):
        self._training_layers[ (block.tile_index, block.block_coords) ] = layer_index

    def get_training_layer_index(self, block: Block ) -> int:
        return self._training_layers.get( (block.tile_index, block.block_coords), -1 )

    @exception_handled
    def epoch_callback(self, epoch):
        if (self.test_mask is not None) and (self.test_size > 0.0):
            prediction = self.predict( self.training_data )
            for iBlock in range( self.training_data.shape[0] ):
                labels = self.training_labels[iBlock].argmax(axis=-1)
                classification_results = np.equal( prediction[iBlock].flatten(), labels )
                test_results = classification_results[ self.test_mask[iBlock] ]
                accuracy = np.count_nonzero( test_results ) / test_results.size
                lgm().log( f"Epoch[{epoch}]-> BLOCK-{iBlock}: Test[{test_results.size}] accuracy: {accuracy:.4f}" )

    def predict( self, data: np.ndarray, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        log = kwargs.get('log',False)
        block: Block = tm().getBlock()
        if log:
            waves = [w.mean() for w in self._model.get_layer(index=0).get_weights()]
            lgm().log(f"SpatialModel[{hex(id(self))}:{hex(id(self._model))}].apply: block={block.block_coords} weights={waves}")
        bshape: List[int] = list(block.shape[1:])
        predictresult: np.ndarray = self._model.predict(data)
        classresult: np.ndarray = predictresult.argmax(axis=-1)
        raster_mask = ~block.raster_mask.flatten()
        lgm().log(f" **** predict: data shape = {data.shape}, predict-result shape = {predictresult.shape}, class-result shape = {classresult.shape},  bshape = {bshape}, raster_mask shape = {raster_mask.shape}")
        for iB in range( classresult.shape[0] ):
            classresult[ iB, raster_mask ] = 0
        result = classresult.reshape( [data.shape[0]] + bshape )
        return result
