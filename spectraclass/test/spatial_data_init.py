from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize( "demo3", 'keelin' )
dm.prepare_inputs()
dm.save_config()

