import traceback

from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

block_size = 150
method = "aec" # "vae"
model_dims = 32
nepochs = 5
niter = 1
reprocess_data = False

try:
    dm: DataManager = DataManager.initialize("img_mgr", 'aviris')
    TileManager.block_size = block_size
    dm.modal.model_dims = model_dims
    dm.modal.reduce_nepoch = nepochs
    dm.modal.reduce_anom_focus = 0.1
    dm.modal.reduce_niter = niter
    dm.modal.reduce_method = method
    dm.modal.refresh_model = False
    dm.modal.modelkey = f"b{block_size}.{method}"
    dm.modal.aeckey = "focus"

    dm.loadCurrentProject()
    dm.modal.initialize_dimension_reduction( refresh=reprocess_data )

except Exception as err:
    traceback.print_exc( )
finally:
    lgm().close()







