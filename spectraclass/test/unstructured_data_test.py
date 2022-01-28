from spectraclass.data.base import DataManager
from spectraclass.gui.points3js import PointCloudManager, pcm

dm: DataManager = DataManager.initialize( "demo4", 'swift' )
dm.loadCurrentProject("main")
dm.save_config()

pcgr = pcm().gui()
