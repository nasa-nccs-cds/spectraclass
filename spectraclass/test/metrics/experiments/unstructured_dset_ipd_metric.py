import numpy as np
from spectraclass.test.metrics.plots import InterpointDistancePlot
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.base import DataManager, dm
np.random.seed(19680801)

dataset =    'swift'   # 'desis'  'aviris'
experiment = "demo4"   # "demo2"  "demo3"
subsample = 500    # 500 for swift, 2000 for tess
nx = 3
ny = 3

metrics =   [   dict( metric = "minkowski", p = 4   ),
                dict( metric = "cosine"             ),
                dict( metric = "minkowski", p = 9   ),
                dict( metric = "jensenshannon"      ),
                dict( metric = "braycurtis"         ),
                dict( metric = "canberra"           ),
                dict( metric = "cityblock"          ),
                dict( metric = "correlation"        ),
                dict( metric = "euclidean"        ) ]

# metrics =   [   dict( metric = "minkowski", p = 9   ),
#                 dict( metric = "braycurtis"         ),
#                 dict( metric = "canberra"           ),
#                 dict( metric = "euclidean"        ) ]

dmg: DataManager = DataManager.initialize( experiment, dataset )
input_data: np.ndarray = dmg.getInputFileData( subsample=subsample )
print( f"Loaded input data, shape = {input_data.shape}" )
dplot = InterpointDistancePlot( nx, ny, data=input_data )

for iy in range(ny):
  for ix in range( nx ):
    mparms = metrics[ ix + iy*nx ]
    dplot.histogram( ix, iy, **mparms )

print( "Plotting..")
dplot.show()
