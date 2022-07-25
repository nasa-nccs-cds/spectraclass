import xarray as xa
import time, traceback, abc, os, copy
import numpy as np
from spectraclass.data.spatial.tile.tile import Block
import scipy, sklearn
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
from .base import LearningModel, KerasModelWrapper

class CNN:

    @classmethod
    def build( cls, name: str, input_shape, nfeatures: int, nclasses: int, **kwargs ) -> models.Sequential:    # (batch, channels, height, width)
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
    def get_sample_weights( cls, labels: np.ndarray, nLabels: int) -> np.ndarray:
        sample_weights: np.ndarray = np.where((labels == 0), 0.0, 1.0)
        label_counts = [np.count_nonzero(labels == iC) for iC in range(nLabels)]
        label_weights = np.array([1.0 / lc if (lc > 0.0) else 0.0 for lc in label_counts])
        weights_sum = label_weights.sum()
        label_weights = {iC: label_weights[iC] / weights_sum for iC in range(nLabels)}
        for iC in range(1, nLabels):
            if label_weights[iC] > 0.0:
                sample_weights[(labels == iC)] = label_weights[iC]
        return np.expand_dims(sample_weights, 0 )

    @classmethod
    def get_block_data(cls) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block: Block = tm().getBlock()
        bdata: xa.DataArray = block.data.transpose('y', 'x', 'band').fillna(0.0)
        return bdata.expand_dims('batch',0)

class CNNLearningModel(KerasModelWrapper):

    def __init__(self, name: str, nfeatures: int ):
        from spectraclass.model.labels import LabelsManager, lm
        bdata = CNN.get_block_data()
        model = CNN.build( name, bdata.squeeze().shape, nfeatures, lm().nLabels )
        KerasModelWrapper.__init__( self, "cnn", model )

    def get_training_set(self, block: Block, **kwargs ) -> Tuple[np.ndarray,np.ndarray]:
        from spectraclass.model.labels import LabelsManager, Action, lm
        label_map: xa.DataArray  = lm().get_label_map( block=block )
        tdata: np.ndarray = block.data.transpose('y', 'x', 'band').fillna(0.0).expand_dims('batch', 0).values
        return ( tdata, label_map.values )

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
        lgm().log(f"Completed learning in {time.time() - t1} sec.")

