from spectraclass.data.base import DataManager, ModeDataManager
from spectraclass.gui.unstructured.application import Spectraclass

app = Spectraclass.instance()
app.configure("spectraclass")

dm: DataManager = DataManager.instance()
mdm: ModeDataManager = dm.mode_data_manager

mdm.model_dims = 16
mdm.subsample = 1

mdm.prepare_inputs()