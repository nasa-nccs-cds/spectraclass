import random, numpy as np, torch
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.base import DataManager
import hvplot.xarray
import holoviews as hv
from spectraclass.learn.svc import SVCLearningModel
import panel as pn
from torch_geometric.data import Data
import xarray as xa

def getMasks( class_data: np.ndarray, num_class_exemplars: int) -> Dict[str, torch.tensor]:
    nclasses: int = class_data.max()
    class_masks: List[np.ndarray] = [(class_data == (iC + 1)) for iC in range(nclasses)]
    test_mask: np.ndarray = (class_data > 0)
    nodata_mask = np.logical_not(test_mask)
    class_indices = [np.argwhere(class_masks[iC]).flatten() for iC in range(nclasses)]
    train_class_indices = [np.random.choice(class_indices[iC], size=num_class_exemplars, replace=False) for iC in
                           range(nclasses)]
    train_indices = np.hstack(train_class_indices)
    train_mask = np.full(test_mask.shape, False, dtype=bool)
    train_mask[train_indices] = True
    test_mask[train_indices] = False
    return dict(train_mask=torch.from_numpy(train_mask),
                test_mask=torch.from_numpy(test_mask),
                nodata_mask=torch.from_numpy(nodata_mask))

nhidden = 32
num_class_exemplars = 5
ntrials = 100
accum_acc = []

dm: DataManager = DataManager.initialize( "indianPines", 'aviris' )
project_data: xa.Dataset = dm.loadCurrentProject( "main" )
feature_data: xa.DataArray = project_data.reduction

for iT in range( ntrials ):
    class_data: np.ndarray = dm.getClassMap().values.flatten().astype(np.compat.long)
    class_masks: Dict[str,torch.tensor] = getMasks( class_data, num_class_exemplars )


    train_mask = class_masks['train_mask']
    test_mask = class_masks['test_mask']
    svc = SVCLearningModel( norm=False )

    X: np.ndarray = feature_data.values[train_mask]
    Y: np.ndarray = class_data[train_mask]
    svc.fit( X, Y )

    xt: np.ndarray = feature_data.values[test_mask]
    yt: np.ndarray = class_data[test_mask]
    yp: np.ndarray = svc.predict( xt )

    acc = np.count_nonzero( yp == yt ) / np.count_nonzero( test_mask )
    print(' Trial[{}]--> Accuracy: {:.4f}'.format(iT,acc))
    accum_acc.append( acc )

print(f' Average Accuracy: {np.array(accum_acc).mean()}')




