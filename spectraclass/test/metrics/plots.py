
import scipy.spatial.distance as sd
import matplotlib.pyplot as plt
import time, numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

class InterpointDistancePlot:

    def __init__( self, nx: int, ny: int, nbins: int, title: str  ) :
        self.nx = nx
        self.ny = ny
        self.nbins = nbins
        self.fig, self.axs = plt.subplots(ny, nx, sharey=True, tight_layout=True)
        self.fig.suptitle( title, fontsize=16 )

    def histogram(self, data: np.ndarray, ix, iy, metric: str, title: str, **kwargs ):
        t0 = time.time()
        distances = sd.pdist( data, metric, **kwargs )
        dmax = distances.max()
        distances = distances / dmax
        self.axs[iy,ix].hist( distances, self.nbins, [0.0, 1.0], True )
        plot_title = f"{title}, mag: {dmax:.2f}, time: {(time.time()-t0):.2f}"
        self.axs[iy,ix].set_title( plot_title, y=1.0, pad=-10, fontsize=10 )
        print(f"Histogram{[ix, iy]}-> {plot_title}")

    def show(self):
        plt.show()
