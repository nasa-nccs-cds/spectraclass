from spectraclass.data.base import DataManager
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm

model_dims = 32
block_size = 150
nepochs = 5
niter = 2
method = "vae"

dm: DataManager = DataManager.initialize("img_mgr", 'aviris')
TileManager.block_size = block_size
AvirisDataManager.model_dims = model_dims
AvirisDataManager.reduce_nepoch = nepochs
AvirisDataManager.reduce_niter = niter
AvirisDataManager.reduce_method = method
AvirisDataManager.modelkey = f"b{block_size}.{method}"

dm.loadCurrentProject()
dm.modal.initialize_dimension_reduction( refresh=True )







