
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import time, numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

class InterpointDistancePlot:

    def __init__( self, nx: int, ny: int, nbins: int = 200, **kwargs  ) :
        self.nx = nx
        self.ny = ny
        self.nbins = nbins
        title = kwargs.get( 'title', "Distribution of inter-point distances" )
        self.data: np.ndarray = kwargs.get( 'data', None )
        self.npoints = kwargs.get( 'npoints', 2000 )
        self.ndim = kwargs.get('ndim', -1)
        self.metric = kwargs.get('metric', None )
        if (self.data is None) and ( self.ndim > 0):
            self.data = ( np.random.rand( self.npoints, self.ndim ) * 2 ) - 1.0
        self.fig, self.axs = plt.subplots(ny, nx, sharey=True, tight_layout=True)
        self.fig.suptitle( title, fontsize=16 )

    def histogram(self,  ix, iy, **kwargs ):
        t0 = time.time()
        title = kwargs.pop( 'title', "" )
        data: np.ndarray= kwargs.pop( 'data', self.data )
        metric: str = kwargs.pop('metric', self.metric )
        assert metric, "Must specify metric"
        if data is None:
            ndim = kwargs.pop( 'ndim', self.ndim )
            assert ndim > 0, "Must specify either 'data' or 'ndim' parameters"
            data = np.random.rand( self.npoints, ndim )
#        input_stats = ( data.min(), data.max(), data.std() )
        distances = sd.pdist( data, metric, **kwargs )
        dmax = distances.max()
        origin = min( distances.min(), 0.0 )
        distances = ( distances - origin) / ( dmax - origin )
        self.axs[iy,ix].hist( distances, self.nbins, [0.0, 1.0], True )
        mpstr = ",".join([ f"{k}={v}" for (k,v) in kwargs.items()])
        plot_title = f"{title} metric: {metric}({mpstr}), ndim: {data.shape[1]}, mag: {dmax:.2f}, time: {(time.time()-t0):.2f}"
        self.axs[iy,ix].set_title( plot_title, y=1.0, pad=-10, fontsize=10 )
        print(f"Histogram{[ix, iy]}-> {plot_title}")

    def show(self):
        plt.show()
