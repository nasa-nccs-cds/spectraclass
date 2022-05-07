from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.tile.tile import Block
import xarray as xa

dm: DataManager = DataManager.initialize("demo1",'keelin')
project_data: xa.Dataset = dm.loadCurrentProject("main")
block: Block = tm().getBlock()

block_data = block.data
reduced_data: xa.DataArray = dm.getModelData().transpose()

dims = [ reduced_data.dims[0], block_data.dims[1], block_data.dims[2] ]
coords = [ (dims[0], reduced_data[dims[0]]), (dims[1], block_data[dims[1]]), (dims[2], block_data[dims[2]]) ]
shape = [ c[1].size for c in coords ]
raster_data = reduced_data.data.reshape( shape )

result = xa.DataArray( raster_data, coords, dims, reduced_data.name,  reduced_data.attrs )
print(".")