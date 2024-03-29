from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.data.spatial.modes import AvirisDataManager
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.gui.spatial.map import MapManager, mm
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
    month = "201908"

    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
    dm.modal.images_glob = f"ang{month}*rfl/ang*_rfl_{version}/ang*_corr_{version}_img"
    AvirisDataManager.version = version
    dm.proc_type = "skl"
    dm.modal.refresh_model = False
    TileManager.block_size = block_size
    TileManager.reprocess = False
    AvirisDataManager.model_dims = model_dims
    AvirisDataManager.reduce_method = method
    AvirisDataManager.modelkey = f"b{block_size}.{method}"
    print( f"Init project {project}, mode = {mode}, modelkey = {AvirisDataManager.modelkey}")

    dm.prepare_inputs()
    classes = [('Class-1', "cyan"),
               ('Class-2', "green"),
               ('Class-3', "magenta"),
               ('Class-4', "blue")]

    lm().setLabels(classes)
    dm.modal.initialize_dimension_reduction()

    maps = mm().gui()


