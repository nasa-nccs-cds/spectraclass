from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass
from gui.SCRAP.points import PointCloudManager

app = Spectraclass.instance()
dm: DataManager = app.configure("demo1",'aviris')
from spectraclass.model.labels import LabelsManager

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]
LabelsManager.instance().setLabels( classes )

gui = PointCloudManager.instance().gui()