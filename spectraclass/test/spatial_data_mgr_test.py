from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass

app = Spectraclass.instance()
dm: DataManager = app.configure("demo1",'aviris')
dm.prepare_inputs()

