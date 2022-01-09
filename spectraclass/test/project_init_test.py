from spectraclass.data.base import DataManager
import matplotlib.pyplot as plt

image_indices = [2,3,4]
dm: DataManager = DataManager.initialize( "demo2", 'desis' )  #  ("demo4", 'swift' ) ( "demo2", 'desis' ) ( "demo2", 'aviris' )
dm.modal.cache_dir = "/Users/tpmaxwel/Development/Cache"
dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/DESIS"
dm.modal.image_names = [ f"DESIS-HSI-L1C-DT0468853252_00{index}-20200628T153803-V0210" for index in image_indices ]
dm.preprocess_data()
dm.loadCurrentProject("main")

#sgui = dm.modal.gui( standalone=True )
#plt.show()