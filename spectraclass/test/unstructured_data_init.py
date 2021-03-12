from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize("demo4", 'tess')
dm.prepare_inputs()
dm.save_config()
