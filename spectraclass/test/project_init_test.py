from spectraclass.data.base import DataManager
import matplotlib.pyplot as plt
load = False

dm: DataManager = DataManager.initialize( "demo2", 'aviris' )
dm.prepare_inputs()

if load:
    dm.loadCurrentProject("main")
    sgui = dm.modal.gui( standalone=True )
    plt.show()