from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass

dm: DataManager = DataManager.initialize( "demo1", 'desis' )
app = Spectraclass.instance()
dm.prepare_inputs()
dm.save_config()

