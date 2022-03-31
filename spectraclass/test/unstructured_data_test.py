from spectraclass.data.base import DataManager
from spectraclass.gui.pointcloud import PointCloudManager, pcm

dm: DataManager = DataManager.initialize( "demo4", 'swift' )
dm.loadCurrentProject("main")
dm.save_config()

pcgr = pcm().gui()
