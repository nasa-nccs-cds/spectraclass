from spectraclass.data.base import DataManager
import matplotlib.pyplot as plt

dm: DataManager = DataManager.initialize( "demo2", 'aviris' )
dm.loadCurrentProject("main")
sgui = dm.modal.gui( standalone=True )
plt.show()