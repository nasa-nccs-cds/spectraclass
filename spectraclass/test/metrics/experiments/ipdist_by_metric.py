import numpy as np
from spectraclass.test.metrics.plots import InterpointDistancePlot
np.random.seed(19680801)

#  ‘braycurtis’, ‘canberra’, ‘chebyshev’,‘correlation’, ‘cosine’, ‘euclidean’
#  ‘jensenshannon’, ‘mahalanobis’, ‘minkowski’, ‘seuclidean’

npoints = 1000
ndim = 32
dim_step = 40
metrics =   [ ("seuclidean", {}), ("cosine", {}), ("minkowski", dict(p=9)), ("jensenshannon", {}),
              ("braycurtis", {}), ("canberra", {}), ("chebyshev", {}), ("correlation", {}), ("mahalanobis", {}) ]
mparms = dict( )
nbins = 200
nx = 3
ny = 3

dplot = InterpointDistancePlot( nx,ny,nbins, f'Distribution of inter-point distances, dimensions = {ndim}' )

data = np.random.rand(npoints, ndim)
for iy in range(ny):
  for ix in range( nx ):
    metric, mparms = metrics[ ix + iy*nx ]
    dplot.histogram( data, ix, iy, metric, f"Metric: {metric} {mparms}", **mparms )

print( "Plotting..")
dplot.show()