from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize( "demo3", 'aviris' )
dm.prepare_inputs()
dm.save_config()

