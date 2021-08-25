import random, numpy as np, torch
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.base import DataManager
import hvplot.xarray
import holoviews as hv
from spectraclass.learn.cnn import CNN
import panel as pn
from torch_geometric.data import Data
import xarray as xa

dm: DataManager = DataManager.initialize("indianPines", 'aviris')

sgd_parms = dict( nepochs = 1000, lr = 0.001, weight_decay = 0.0005, dropout = True )
ntrials = 40
num_class_exemplars = 5
accuracy = []

X: torch.Tensor = CNN.getConvData(dm)
class_data: np.ndarray = dm.getClassMap().values
Y: torch.Tensor = torch.from_numpy( class_data.flatten().astype(np.compat.long) ) - 1
Nf = X.shape[1]
Nc = class_data.max()
LS = [ Nf, 25 ]
KS = [ 5, 3 ]

for iT in range(ntrials):
    masks = CNN.getMasks( class_data, num_class_exemplars )
    train_data = Data(x=X, y=Y, **masks)
    cnn = CNN( LS, Nc, KS )
    cnn.train_model( train_data, **sgd_parms )
    (pred, acc) = cnn.evaluate_model( train_data )
    accuracy.append( acc )
    print(f" ** Completed trial {iT}/{ntrials}: Accuracy = {acc}, running average = {np.array(accuracy).mean()}")

acc_data = np.array(accuracy)
print( f"Average accuracy over {ntrials} trials = {acc_data.mean()}, std = {acc_data.std()}")




