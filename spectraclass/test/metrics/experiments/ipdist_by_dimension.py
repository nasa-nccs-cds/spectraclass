import numpy as np
from spectraclass.test.metrics.plots import InterpointDistancePlot
np.random.seed(19680801)

npoints = 2000
min_dim = 3
dim_step = 40
metric = "cosine"
mparms = dict( )
nbins = 200
nx = 3
ny = 3

dplot = InterpointDistancePlot( nx,ny,nbins, metric=metric, npoints=npoints )

for iy in range(ny):
  for ix in range( nx ):
    ndim = min_dim + ( ix + iy*nx ) * dim_step
    dplot.histogram( ix, iy, ndim=ndim )

print( "Plotting..")
dplot.show()
