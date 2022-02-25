from spectraclass.data.base import DataManager
from spectraclass.data.base import ModeDataManager
from spectraclass.model.labels import LabelsManager, lm

dm: DataManager = DataManager.initialize( "demo4", 'tess' )
dm.modal.cache_dir = "/Volumes/Shared/Cache"
dm.modal.data_dir = "/Volumes/Shared/Data/tess"
dm.proc_type = "cpu"
ModeDataManager.model_dims = 24

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

dm.loadCurrentProject()
labels_data = lm().getLabelsArray()