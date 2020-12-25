from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass

app = Spectraclass.instance()
dm: DataManager = app.configure("demo1",'desis')
from spectraclass.model.labels import LabelsManager

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]
LabelsManager.instance().setLabels( classes )

app.gui( False )