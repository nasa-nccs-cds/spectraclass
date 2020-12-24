from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass

app = Spectraclass.instance()
dm: DataManager = app.configure("demo1",'desis')
dm.prepare_inputs()
app.save_config()

