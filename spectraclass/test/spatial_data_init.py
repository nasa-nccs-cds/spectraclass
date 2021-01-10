from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass

dm: DataManager = DataManager.initialize( "demo2", 'desis' )
app = Spectraclass.instance()
dm.modal.reduce_scope = "tile"
dm.prepare_inputs()
dm.save_config()

