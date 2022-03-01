from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.gui.unstructured.table import tbm

dm: DataManager = DataManager.initialize( "demo4", 'swift' )
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Volumes/Shared/Data/tess"
dm.proc_type = "cpu"
ModeDataManager.model_dims = 24

project = dm.loadCurrentProject()
gui = tbm().gui()
print("")
