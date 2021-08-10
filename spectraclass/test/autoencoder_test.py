from spectraclass.data.base import DataManager


dm: DataManager = DataManager.initialize( "demo2", 'desis' )  #  ("demo4", 'swift' ) ( "demo2", 'desis' )
dm.loadCurrentProject("main")
dm.prepare_inputs()

