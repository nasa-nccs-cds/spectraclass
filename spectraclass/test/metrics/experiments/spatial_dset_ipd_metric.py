import numpy as np
from spectraclass.test.metrics.plots import InterpointDistancePlot
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.base import DataManager, dm
np.random.seed(19680801)

dataset = 'aviris'     #  'desis'      aviris
experiment = "demo3"   #  "demo2"     demo3
subsample = 100        #   100         100
nx = 3
ny = 3

metrics =   [   dict( metric = "minkowski", p = 4   ),
                dict( metric = "cosine"             ),
                dict( metric = "minkowski", p = 9   ),
                dict( metric = "seuclidean"         ),
                dict( metric = "braycurtis"         ),
                dict( metric = "canberra"           ),
                dict( metric = "cityblock"          ),
                dict( metric = "correlation"        ),
                dict( metric = "euclidean"        ) ]

dmg: DataManager = DataManager.initialize( experiment, dataset )
block = tm().getBlock()
( xa_point_data, xa_point_coords ) = block.getPointData( subsample=subsample )

print( f"Loaded point data, shape = {xa_point_data.data.shape}")
dplot = InterpointDistancePlot( nx, ny, data=xa_point_data.data, title=f"Distribution of inter-point distances for {dataset.upper()} data" )

for iy in range(ny):
  for ix in range( nx ):
    mparms = metrics[ ix + iy*nx ]
    dplot.histogram( ix, iy, **mparms )

print( "Plotting..")
dplot.show()
