from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize( "demo1", 'keelin' )
dm.prepare_inputs()
dm.save_config()

