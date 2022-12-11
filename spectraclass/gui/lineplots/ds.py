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
        self._curves = {}

    def gui(self):
        return self.plot

    def clearTransients( self ):
        pass

    @log_timing
    @exception_handled
    def addMarker( self, m: Marker ):
        lgm().log(f"mplGraphPlot: Add Marker[{m.size}]: cid={m.cid}, pids[:10]={m.gids[:10]}")
        if m.size > 0:
            self.clearTransients()
            self._markers.append( m )
            for pid in m.gids:
                self._curves[pid] = hv.Curve( self.ly(pid) )
            self.plot()

    @property
    def pids(self) -> List[int]:
        rv = []
        for m in self._markers:
            rv.extend(m.gids)
        return rv

    @property
    def tpids(self) -> List[int]:
        rv = []
        for m in self._markers:
            if m.cid == 0: rv.extend(m.gids)
        return rv

    @log_timing
    def plot(self):
        self.overlay = hv.NdOverlay( self._curves )
        self.plot = datashade( self.overlay, normalization='linear', aggregator=ds.count() ).opts(self.opts)
        lgm().log( f"dsGraphPlot->plot: {self.plot.__class__}")



