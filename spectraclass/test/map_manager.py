from spectraclass.gui.spatial.map import MapManager, mm
from spectraclass.data.base import DataManager
import sys, matplotlib.pyplot as plt
from spectraclass.model.labels import LabelsManager, lm
def iname( indx ): return f"DESIS-HSI-L1C-DT0468853252_00{indx}-20200628T153803-V0210"

image_indices = [2,3,4]
dm: DataManager = DataManager.initialize( "demo2", 'desis' )  #  ("demo4", 'swift' ) ( "demo2", 'desis' ) ( "demo2", 'aviris' )
dm.modal.cache_dir = "/Users/tpmaxwel/Development/Cache"
dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/DESIS"
dm.modal.image_names = [ iname(index) for index in image_indices ]
dm.loadCurrentProject("main")

def on_press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key in [ str(ii) for ii in image_indices ]:
        dm.modal.set_image_name( iname( event.key ) )

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]
lm().setLabels( classes )

map_canvas = mm().gui( standalone=True, parallel=False )
map_canvas.mpl_connect('key_press_event', on_press)
print( "Generated canvas" )


plt.show( block=True )