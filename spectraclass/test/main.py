from spectraclass.data.base import DataManager
from spectraclass.application.controller import app, SpectraclassController
from spectraclass.model.labels import LabelsManager, lm

dm: DataManager = DataManager.initialize( "demo2", 'aviris' )  #  ("demo4", 'swift' ) ( "demo2", 'desis' ) ( "demo2", 'aviris' )
dm.loadCurrentProject("main")

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

lm().setLabels( classes )
controller: SpectraclassController = app()
controller.gui()