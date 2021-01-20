from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize( "demo2", 'desis' )
dm.modal.reduce_scope = "tile"
dm.prepare_inputs()
dm.save_config()

