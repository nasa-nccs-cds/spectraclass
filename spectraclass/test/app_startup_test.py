from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm

dm: DataManager = DataManager.initialize("demo4",'keelin')
dm.loadCurrentProject("main")

block = tm().getBlock()
print(f"Block data shape = {block.data.shape}")