from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm

dm: DataManager = DataManager.initialize("demo2",'keelin')
dm.loadCurrentProject("main")

block = tm().getBlock()

point_data, point_coords = block.getPointData()
bounds = [ block.xlim, block.ylim ]

print(f"Block data shape = {block.data.shape}")