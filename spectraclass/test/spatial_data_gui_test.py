import matplotlib
matplotlib.rcParams['toolbar'] = 'toolmanager'
from spectraclass.data.base import DataManager
from spectraclass.gui.points import PointCloudManager, pcm
from spectraclass.gui.spatial.map import MapManager, mm
from spectraclass.model.labels import LabelsManager, lm

dm: DataManager = DataManager.initialize("demo2", 'desis' )
dm.loadCurrentProject("main")

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

lm().setLabels( classes )

pcgr = pcm().gui()