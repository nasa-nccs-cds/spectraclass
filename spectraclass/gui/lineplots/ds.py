import datetime
import pandas as pd
import numpy as np
import xarray
import xarray as xr
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from spectraclass.util.logs import LogManager, lgm
import datashader as ds
import datashader.transfer_functions as tf
from collections import OrderedDict
from .manager import LinePlot
import holoviews as hv
from holoviews.operation.datashader import datashade
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
hv.extension('bokeh')

class dsGraphPlot(LinePlot):

    def __init__( self, index: int, **kwargs ):
        LinePlot.__init__( self, index, **kwargs )
        self.opts = hv.opts.RGB(width=800, height=400)
        self.overlay = hv.NdOverlay({})
        self.plot = datashade( self.overlay, normalization='linear', aggregator=ds.count() ).opts(self.opts)
        self._markers = []

    def gui(self):
        return

    def clearTransients( self ):
        pass

    @log_timing
    @exception_handled
    def addMarker( self, m: Marker ):
        lgm().log(f"mplGraphPlot: Add Marker[{m.size}]: cid={m.cid}, pids[:10]={m.pids[:10]}")
        if m.size > 0:
            self.clearTransients()
            self._markers.append( m )
            self.plot()

    @property
    def pids(self) -> List[int]:
        pass

    @property
    def tpids(self) -> List[int]:
        pass

    @log_timing
    def plot(self):
        from spectraclass.model.labels import LabelsManager, lm
        color = lm().graph_colors[cid]
        lrecs = [ LineRec(None, pid, cid) for pid in pids ]
        for lrec in lrecs: self.lrecs[lrec.pid] = lrec
        lines = self.ax.plot( self.lx(pids), self.ly(pids), picker=True, pickradius=2, color=color, alpha=1.0, linewidth=1.0 )
        self.ax.figure.canvas.draw_idle()
        for (lrec, line) in zip(lrecs, lines): lrec.line = line


#        self.df = pd.DataFrame( self._ploty )
#        lgm().log( self.df.head(5) )

#agg = cvs.line(df, x=time, y=list(range(points)), agg=ds.count(), axis=1)
#img = tf.shade(agg, how='eq_hist')