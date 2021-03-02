from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.gui.plot import JbkPlot

dm: DataManager = DataManager.initialize("demo4",'keelin')
dm.loadCurrentProject("main")

block = tm().getBlock()
print(f"Block data shape = {block.data.shape}")

graph0 = JbkPlot()
graph0.select_items( [ 1000 ] )
graph0.plot()

print("DOME")