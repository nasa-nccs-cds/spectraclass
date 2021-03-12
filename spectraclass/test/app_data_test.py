import matplotlib
matplotlib.rcParams['toolbar'] = 'toolmanager'
from spectraclass.data.base import DataManager
from spectraclass.gui.plot import GraphPlotManager, gpm
from spectraclass.model.labels import LabelsManager, lm

dm: DataManager = DataManager.initialize( "demo4", 'tess' )
dm.loadCurrentProject("main")

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

lm().setLabels( classes )

gui = gpm().gui()

