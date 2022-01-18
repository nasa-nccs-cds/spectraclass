import xarray as xa
import time, traceback, abc
import numpy as np
import scipy, sklearn
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
from ..model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
import ipywidgets as ipw
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

class LearningModel:

    def __init__(self, name: str,  **kwargs ):
        self.mid =  name
        self._score: Optional[np.ndarray] = None
        self.config = kwargs
        self._keys = []

    def setKeys(self, keys: List[str] ):
        self._keys = keys

    @property
    def score(self) -> Optional[np.ndarray]:
        return self._score

    def learn_classification( self, point_data: np.ndarray, labels: np.ndarray, **kwargs  ):
        t1 = time.time()
        if np.count_nonzero( labels > 0 ) == 0:
            ufm().show( "Must label some points before learning the classification" )
            return None
        self.fit( point_data, labels, **kwargs )
        print(f"Learned mapping with {labels.shape[0]} labels in {time.time()-t1} sec.")

    def fit(self, data: np.ndarray, labels: np.ndarray, **kwargs):
        raise Exception( "abstract method LearningModel.fit called")

    def apply_classification( self, data: xa.DataArray, **kwargs ) -> xa.DataArray:
        t1 = time.time()
        prediction: np.ndarray = self.predict( data.values, **kwargs )
        print(f"Applied classication with input shape {data.shape[0]} in {time.time() - t1} sec.")
        return xa.DataArray( prediction, dims=['samples'], coords=dict( samples=data.coords['samples'] ) )

    def predict( self, data: np.ndarray, **kwargs ):
        raise Exception( "abstract method LearningModel.predict called")


class LearningModelWrapper(LearningModel):

    def __init__(self, name: str,  model: Model, **kwargs ):
        LearningModel.__init__( self, name,  **kwargs )
        self._model = model
        self.optimizer = self.get_optimizer( **kwargs )
        self.criterion = CrossEntropyLoss()
        if torch.cuda.is_available():
            self._model = self._model.cuda()
            self.criterion = self.criterion.cuda()

    def get_optimizer( self, **kwargs ):
        opt = str(kwargs.get( 'opt', 'SGD' )).lower()
        parms = self._model.parameters()
        if opt == "sgd": return SGD( parms, lr=0.07)
        elif opt == "adam": return Adam(parms, lr=0.07)
        else: raise Exception( f"Unknown optimizer: {opt}" )

    def predict( self, data: np.ndarray, **kwargs ):
        raise Exception( "abstract method LearningModel.predict called")


    def fit(self, data: np.ndarray, labels: np.ndarray, **kwargs):
        raise Exception( "abstract method LearningModel.fit called")

    def train( self, epoch ):
        self._model.train()
        tr_loss = 0
        x_train, y_train = Variable(train_x), Variable(train_y)
        # getting the validation set
        x_val, y_val = Variable(val_x), Variable(val_y)
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()
        self.optimizer.zero_grad()
        output_train = self._model(x_train)
        output_val = self._model(x_val)
        loss_train = self.criterion(output_train, y_train)
        loss_val = self.criterion(output_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        loss_train.backward()
        self.optimizer.step()
        tr_loss = loss_train.item()
        if epoch % 2 == 0:
            print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)



