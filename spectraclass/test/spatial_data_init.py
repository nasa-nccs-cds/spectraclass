from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize( "demo3", 'aviris' )
dm.modal.reduce_scope = "tile"
dm.prepare_inputs()
dm.save_config()

