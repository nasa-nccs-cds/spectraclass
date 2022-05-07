from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize( "demo4", 'swift' )
dm.prepare_inputs()
dm.save_config()

