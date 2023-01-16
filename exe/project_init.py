from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.data.spatial.modes import AvirisDataManager
import sys

if len(sys.argv) != 3:
    print( f"Usage: {sys.argv[0]} <mode> <project>")
else:

    mode: str = sys.argv[1]       #   e.g. 'swift', 'tess', 'desis', or 'aviris'
    project: str = sys.argv[2]    #   e.g. 'demo1', 'demo2', 'demo3', or 'demo4'
    dm: DataManager = DataManager.initialize(project, mode)

    block_size = 200
    method = "aec" # "aec" "vae"
    model_dims = 30
    version = "v2p9"
    month = "201908"

    dm.modal.images_glob = f"ang{month}*rfl/ang*_rfl_{version}/ang*_corr_{version}_img"
    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
    AvirisDataManager.version = version
    dm.proc_type = "cpu"
    dm.modal.refresh_model = False
    TileManager.block_size = block_size
    TileManager.reprocess = False
    AvirisDataManager.model_dims = model_dims
    AvirisDataManager.valid_aviris_bands = [ [0, 10000], ]
    AvirisDataManager.reduce_method = method
    AvirisDataManager.modelkey = f"b{block_size}.{method}.{model_dims}"
    print( f"Init project {project}, mode = {mode}, modelkey = {AvirisDataManager.modelkey}")

    dm.prepare_inputs()




