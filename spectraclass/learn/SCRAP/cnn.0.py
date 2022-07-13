import xarray as xa
import time, traceback, abc
from sklearn.model_selection import train_test_split
import numpy as np
import scipy, sklearn
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional, Dict
from model.labels import LabelsManager
import traitlets as tl
import traitlets.config as tlc
import ipywidgets as ipw
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import torch, time
import torch.nn.functional as F
from learn.base import LearningModel
import xarray as xa
import numpy as np
from spectraclass.data.base import DataManager
from typing import List, Union, Tuple, Optional, Dict
def lname( iL: int ): return f"layer-{iL}"

class CNN:

    @classmethod
    def build( cls, input_shape: List[int], nclasses: int, **kwargs ) -> models.Sequential:    # (batch, channels, height, width)
        ks = kwargs.get('kernel_size',3)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (ks,ks), activation='relu', input_shape=input_shape, data_format="channels_first"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten(data_format="channels_first") )
        model.add(layers.Dense(64, activation='relu'))
        model.add( layers.Dense( nclasses, activation='linear' ) )
        activation = tf.keras.layers.Softmax( axis=-1, **kwargs )
        model.add( layers.Dense( nclasses, activation=activation ))
        return model




class Data():

    def __init__(self, x, y, masks: Dict ):
        self.x = x
        self.y = y
        self.train_mask = masks.get('train_mask')
        self.test_mask = masks.get('test_mask')
        self.nodata_mask = masks.get('nodata_mask')

class CNNLearningModel(LearningModel):

    def __init__(self, nnet: torch.nn.Module, **kwargs ):
        LearningModel.__init__(self, "svc",  **kwargs )
        self._score: Optional[np.ndarray] = None
        self._nnet = nnet
        self.spectral_graphs = {}
        self.spatial_graphs = {}

    def getTrainingData( self, **kwargs ):       # X[n_samples, n_features], y[n_samples]
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.data.base import DataManager, dm
        t0 = time.time()
        X: torch.Tensor = self.getReducedDataGrid(dm())
        class_data: xa.DataArray = lm().getLabelsArray()
        Y: torch.Tensor = torch.from_numpy(class_data.flatten().astype(np.compat.long)) - 1
        train_mask: torch.tensor = torch.from_numpy( class_data.values > 0 )
        test_mask:  torch.tensor = torch.from_numpy( class_data.values == 0 )
        nodata_mask:  torch.tensor = torch.from_numpy( class_data.values == class_data.attrs['_FillValue'] )
        class_masks: Dict[str, torch.tensor] = dict( train_mask=train_mask, test_mask=test_mask, nodata_mask=nodata_mask)
        input_data = Data( X, Y, class_masks)
        self.train_model( input_data, **kwargs )
        print(f"Completed NN fit, in {time.time()-t0} secs")


    def predict( self, X: np.ndarray, **kwargs ) -> np.ndarray:
        input_data = self.getTrainingData(X)
        (pred, acc) = self._nnet.evaluate_model(input_data)
        return pred.numpy()

    def getMasks(self, class_data: np.ndarray, num_class_exemplars: int) -> Dict[str, torch.tensor]:
        nclasses: int = class_data.max()
        class_masks: List[np.ndarray] = [(class_data == (iC + 1)) for iC in range(nclasses)]
        test_mask: np.ndarray = (class_data > 0)
        nodata_mask = np.logical_not(test_mask)
        class_indices = [np.argwhere(class_masks[iC]).flatten() for iC in range(nclasses)]
        train_class_indices = [ np.random.choice(class_indices[iC], size=num_class_exemplars, replace=False) for iC in range(nclasses) ]
        train_indices = np.hstack(train_class_indices)
        train_mask = np.full(test_mask.shape, False, dtype=bool)
        train_mask[train_indices] = True
        test_mask[train_indices] = False
        return dict(train_mask=torch.from_numpy(train_mask),
                    test_mask=torch.from_numpy(test_mask),
                    nodata_mask=torch.from_numpy(nodata_mask))

    def getTrainingData(self, num_class_exemplars) -> Data:
        from spectraclass.data.base import DataManager, dm
        X: torch.Tensor = self.getReducedDataGrid(dm())
        class_map: xa.DataArray = dm().getClassMap()
        class_data: np.ndarray = class_map.values
        Y: torch.Tensor = torch.from_numpy(class_data.flatten().astype(np.compat.long)) - 1
        masks = self.getMasks( class_data, num_class_exemplars )
        return Data( X, Y, masks )

    def train_model( self, data: Data, **kwargs ):
        lr = kwargs.get('lr',0.01)
        weight_decay = kwargs.get('weight_decay', 5e-4)
        nepochs = kwargs.get( 'nepochs', 200 )
        self._dropout = kwargs.get( 'dropout', True )
        optimizer = torch.optim.Adam( self._nnet.parameters(), lr=lr, weight_decay=weight_decay )
        self._nnet.train()
        for epoch in range(nepochs):
            optimizer.zero_grad()
            out = self._nnet(data)
            loss = F.nll_loss( out[data.train_mask], data.y[data.train_mask] )
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'epoch: {epoch}, loss = {loss.data}' )

    def npnorm(self, data: np.ndarray, axis: int ):
        v0, v1 = data.min(axis=axis).elem(), data.max(axis=axis).elem()
        return ( data - v0 ) / ( v1 - v0 )

    def evaluate_model( self, data: Data ) -> Tuple:
        self._nnet.eval()
        class_results = self._nnet(data)
        rvals, pred = class_results.max(dim=1)
        reliability = rvals.detach().numpy()
        correct = int( pred[data.test_mask].eq( data.y[data.test_mask] ).sum().item() )
        acc = correct / int( data.test_mask.sum() )
        pred_data = pred.numpy() + 1
        pred_data[ data.nodata_mask.numpy() ] = 0
        print( f"Class prob range = ({class_results.min()}, {class_results.max()}), mean = {class_results.mean()}, std = {class_results.std()}")
        return ( pred_data, reliability, acc )

    @staticmethod
    def getReducedDataGrid(dm: DataManager) -> torch.Tensor:
        project_data: xa.Dataset = dm.loadCurrentProject("main")
        reduced_spectral_data: xa.DataArray = project_data['reduction']
        raw_spectral_data: xa.DataArray = project_data['raw']
        Nf = reduced_spectral_data.shape[1]
        Nw = raw_spectral_data.shape[2]
        Nh = raw_spectral_data.shape[1]
        X = reduced_spectral_data.values.transpose().reshape([1, Nf, Nh, Nw])
        return torch.from_numpy(X)
