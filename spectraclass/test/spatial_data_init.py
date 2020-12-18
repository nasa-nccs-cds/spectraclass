from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass
from spectraclass.model.labels import LabelsManager

app = Spectraclass.instance()
dm: DataManager = app.initialize("demo1",'aviris')


