from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.base import DataManager
import shapely.vectorized as svect
from shapely.geometry import Polygon
from spectraclass.xext.xgeo import XGeo
import numpy as np
import xarray as xa

dm: DataManager = DataManager.initialize( "demo2", 'desis' )
raster: xa.DataArray = tm().getBlock().data[100,:100,:].squeeze()
X,Y = raster.x.values, raster.y.values
shape = raster.shape
dims = raster.dims

poly_xy = [ [-8543685.01133318, 4736359.626343201],
            [-8543116.96631656, 4737902.645638049],
            [-8540844.78625008, 4737547.928558773],
            [-8541707.37312717, 4735863.022432216],
            [-8543685.01133318, 4736359.626343201] ]
polygon = Polygon( poly_xy )

MX, MY = np.meshgrid( X, Y )
PID = np.array( range( raster.size ) )
mask = svect.contains( polygon, MX, MY )
pids = PID[ mask.flatten() ].tolist()
print( pids )
