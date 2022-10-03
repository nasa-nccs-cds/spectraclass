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

    block_size = 150
    method = "aec" # "aec" "vae"
    model_dims = 32
    version = "v2v2"

    dm.modal.images_glob = f"ang201908*rfl/ang*_rfl_{version}/ang*_corr_{version}_img"
    dm.proc_type = "skl"
    dm.modal.refresh_model = True
    TileManager.block_size = block_size
    TileManager.reprocess = False
    AvirisDataManager.model_dims = model_dims
    AvirisDataManager.version = version
    AvirisDataManager.reduce_method = method
    AvirisDataManager.modelkey = f"b{block_size}.{method}"
    print( f"Init project {project}, mode = {mode}, modelkey = {AvirisDataManager.modelkey}")

    dm.prepare_inputs()


