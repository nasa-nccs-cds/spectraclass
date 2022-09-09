import matplotlib.pyplot as plt
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.tile import Block, Tile
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.data.spatial.tile.manager import TileManager, tm
import xarray as xa, numpy as np

mode: str = 'aviris'
project: str = "img_mgr"

dm: DataManager = DataManager.initialize(project, mode)
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
dm.modal.refresh_model = True

block_size = 150
method = "vae" # "vae" "aec"
model_dims = 32
subsample = 100

activation= "relu" # "relu" # "tanh"
optimizer= 'rmsprop' # 'adam' 'sgd' 'rmsprop'
learning_rate = .0002
dropout = 0.005
regularizer = 0.002
niter = 5
nepoch = 5
vaargs= dict( activation=activation, optimizer=optimizer, dropout=dropout, regularizer=regularizer, learning_rate=learning_rate )

dm.modal.ext = "_img"
dm.proc_type = "skl"
TileManager.block_size = block_size
TileManager.block_index = (0,7)
AvirisDataManager.model_dims = model_dims
AvirisDataManager.reduce_method = method
AvirisDataManager.reduce_niter = niter
AvirisDataManager.reduce_nepoch = nepoch
AvirisDataManager.modelkey = f"b{block_size}.{method}"
print( f"Init project {project}, mode = {mode}, modelkey = {AvirisDataManager.modelkey}")

dm.loadCurrentProject()
classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

lm().setLabels( classes )
dm.modal.initialize_dimension_reduction( target_block=TileManager.block_index, **vaargs )

block: Block = tm().getBlock()
point_data: xa.DataArray = block.getPointData()[0]
model_data: xa.DataArray = block.model_data
reproduction: xa.DataArray = block.reproduction
diff = np.abs(point_data.values - reproduction.values)
x = point_data.coords['band'].values
print( f"reproduction shape = {reproduction.shape}, dims={reproduction.dims}, ave diff = {diff.mean()}")

input_data = point_data.values[::subsample].transpose()
result_data = reproduction.values[::subsample].transpose()
diff_data = diff[::subsample].transpose()

fig0, ax0 = plt.subplots( 1, 3, sharex='all', sharey='all' )
ax0[0].plot( x, input_data, alpha = 0.1, lw=1, color="green" )
ax0[1].plot( x, result_data, alpha = 0.1, lw=1, color="blue"  )
ax0[2].plot( x, diff_data, alpha = 0.1, lw=1, color="red"  )

plt.show()




