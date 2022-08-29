import traceback

from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

model_dims = 16
block_size = 150
nepochs = 5
niter = 1
method = "vae"

try:
    dm: DataManager = DataManager.initialize("img_mgr", 'aviris')
    TileManager.block_size = block_size
    dm.modal.model_dims = model_dims
    dm.modal.reduce_nepoch = nepochs
    dm.modal.reduce_niter = niter
    dm.modal.reduce_method = method
    dm.modal.modelkey = f"b{block_size}.{method}"

    dm.loadCurrentProject()
    dm.modal.initialize_dimension_reduction( refresh=True )
except Exception as err:
    traceback.print_exc( )
finally:
    lgm().close()







