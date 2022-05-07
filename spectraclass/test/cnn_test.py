import numpy as np, torch
from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.image import plot_results, toXA
from learn.SCRAP.cnn import CNN
from torch_geometric.data import Data
import xarray as xa

dm: DataManager = DataManager.initialize("pavia", 'aviris')

sgd_parms = dict( nepochs = 100, lr = 0.005, weight_decay = 0.0005, dropout = True )
ntrials = 1
nHidden = 32
num_class_exemplars = 5
iFeature = 0
accuracy = []
best_model = None
best_acc = 0.0
best_data = None

X: torch.Tensor = CNN.getInputData(dm)
class_map: xa.DataArray = dm.getClassMap()
class_data: np.ndarray = class_map.values
nodata_mask = (class_data.flatten() == 0)
Y: torch.Tensor = torch.from_numpy( class_data.flatten().astype(np.compat.long) ) - 1
Nf = X.shape[1]
Nc = class_data.max()
LS = [ Nf, nHidden ]
KS = [ 3, 3 ]

for iT in range(ntrials):
    masks = CNN.getMasks( class_data, num_class_exemplars )
    train_data = Data(x=X, y=Y, **masks)
    cnn = CNN( LS, Nc, KS )
    cnn.train_model( train_data, **sgd_parms )
    (pred, reliability, acc) = cnn.evaluate_model( train_data )
    accuracy.append( acc )
    print(f" ** Completed trial {iT}/{ntrials}: Accuracy = {acc}, running average = {np.array(accuracy).mean()}")
    if acc > best_acc:
        best_acc = acc
        best_model = cnn
        best_data = train_data


acc_data = np.array(accuracy)
print( f"Average accuracy over {ntrials} trials = {acc_data.mean()}, std = {acc_data.std()}")

( pred, reliability, acc ) = best_model.evaluate_model( best_data )
ntest  = np.count_nonzero( best_data.test_mask.numpy() )
ntrain = np.count_nonzero( best_data.train_mask.numpy() )
print( f"Plotting classification, accuracy = {acc}, with {ntest} test labels and {ntrain} train labels ({Nc} classes)")
pred_class_map: xa.DataArray = class_map.copy( data = pred.reshape( class_map.shape ) )
feature_map = toXA( "featureMap", X.numpy().squeeze() )
reliability[nodata_mask] = 0.0
reliability_map = toXA( "reliability", reliability.reshape(class_map.shape) )
plot_results( class_map, pred_class_map, feature_map, reliability_map )




