import random, numpy as np, torch
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.base import DataManager
import hvplot.xarray
import holoviews as hv
from spectraclass.learn.svc import SVCLearningModel
import panel as pn
from spectraclass.gui.spatial.image import toXA
import xarray as xa

def plot_results( class_map: xa.DataArray, pred_class_map: xa.DataArray, reliability_map: xa.DataArray ):
    class_plot = class_map.hvplot.image(cmap='Category20')
    pred_class_plot = pred_class_map.hvplot.image( cmap='Category20' )
#    kwargs = {} if (feature_maps.ndim < 3) else dict( groupby=feature_maps.dims[0], widget_type='scrubber', widget_location='bottom' )
#    feature_plot = feature_maps.hvplot.image( cmap='jet', **kwargs )
    reliability_plot = reliability_map.hvplot.image(cmap='jet')
    pn.Row( pn.Column( class_plot, pred_class_plot ), pn.Column( reliability_plot) ).show( str(class_map.name) )


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

dm: DataManager = DataManager.initialize( "pavia", 'aviris' )
project_data: xa.Dataset = dm.loadCurrentProject( "main" )
class_map: xa.DataArray = dm.getClassMap()
class_data: np.ndarray = class_map.values.flatten().astype(np.compat.long)
feature_data: xa.DataArray = project_data['reduction']

ntrials = 10
nHidden = 32
num_class_exemplars = 10

class_masks: Dict[str,torch.tensor] = getMasks( class_data, num_class_exemplars )
train_mask = class_masks['train_mask']
test_mask = class_masks['test_mask']
nodata_mask= class_masks['nodata_mask']
svc = SVCLearningModel( norm=False )

X: np.ndarray = feature_data.values[train_mask]
Y: np.ndarray = class_data[train_mask]
svc.fit( X, Y )

xt: np.ndarray = feature_data.values[test_mask]
yt: np.ndarray = class_data[test_mask]
yp: np.ndarray = svc.predict( xt )

acc = np.count_nonzero( yp == yt ) / np.count_nonzero( test_mask )
print(' Accuracy: {:.4f}'.format(acc))

probability_map: np.ndarray = svc.probability( xt )
ypp = probability_map[ np.arange(probability_map.shape[0]), yp-1 ]
reliability_data = np.zeros( class_map.shape ).flatten()
reliability_data[test_mask] = ypp
reliability_data[nodata_mask] = 0

result_data: np.ndarray = np.zeros( class_data.shape )
result_data[ test_mask ] = yp
classification_map: xa.DataArray = toXA( 'classification_map', result_data.reshape( class_map.shape ) )
reliability_map = toXA( 'reliability_map', reliability_data.reshape( class_map.shape ) )
feature_map = toXA( 'feature_map', feature_data.values.transpose().reshape(  feature_data.shape[1:] + class_map.shape ) )
plot_results( class_map, classification_map, reliability_map )





