from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize( "demo2", 'desis' )
dm.prepare_inputs()
dm.save_config()

