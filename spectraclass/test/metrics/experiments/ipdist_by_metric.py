import numpy as np
from spectraclass.test.metrics.plots import InterpointDistancePlot
np.random.seed(19680801)

#  ‘braycurtis’, ‘canberra’, ‘chebyshev’,‘correlation’, ‘cosine’, ‘euclidean’
#  ‘jensenshannon’, ‘mahalanobis’, ‘minkowski’, ‘seuclidean’

npoints = 1000
ndim = 400
dim_step = 40
metrics =   [   dict( metric = "seuclidean"         ),
                dict( metric = "cosine"             ),
                dict( metric = "minkowski", p = 9   ),
                dict( metric = "jensenshannon"      ),
                dict( metric = "braycurtis"         ),
                dict( metric = "canberra"           ),
                dict( metric = "chebyshev"          ),
                dict( metric = "correlation"        ),
                dict( metric = "mahalanobis"        ) ]

nbins = 200
nx = 3
ny = 3
dplot = InterpointDistancePlot( nx, ny, nbins, ndim=ndim )

for iy in range(ny):
  for ix in range( nx ):
    mparms = metrics[ ix + iy*nx ]
    dplot.histogram( ix, iy, **mparms )

print( "Plotting..")
dplot.show()