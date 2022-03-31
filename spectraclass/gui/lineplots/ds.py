import datetime
import pandas as pd
import numpy as np
import xarray
import xarray as xr
import ipywidgets as ipw
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from spectraclass.util.logs import LogManager, lgm
import datashader as ds
import datashader.transfer_functions as tf
from collections import OrderedDict
from .manager import LinePlot
import holoviews as hv
import panel as pn
from holoviews.operation.datashader import datashade
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
hv.extension('bokeh')

class dsGraphPlot(LinePlot):

    def __init__( self, index: int, **kwargs ):
        LinePlot.__init__( self, index, **kwargs )
        self.opts = hv.opts.RGB(width=800, height=400)
        self._overlay = hv.NdOverlay({})
        self._plot = datashade(self._overlay, normalization='linear', aggregator=ds.count()).opts(self.opts)
        self._markers = []
        self._curves = {}

    def clear(self):
        self._curves = {}
        self.plot()

    def remove_region(self, m: Marker):
        for pid in m.pids: self._curves.pop( pid )
        self.plot()

    def remove_point(self, pid: int ):
        self._curves.pop( pid )
        self.plot()

    def gui(self):
        return ipw.HBox( [ hv.render(self._plot) ] )

    def clearTransients( self ):
        pass

    @log_timing
    @exception_handled
    def addMarker( self, m: Marker ):
        lgm().log(f"mplGraphPlot: Add Marker[{m.size}]: cid={m.cid}, pids[:10]={m.pids[:10]}")
        if m.size > 0:
            self.clearTransients()
            self._markers.append( m )
            for pid in m.pids:
                self._curves[pid] = hv.Curve( self.ly(pid) )
            self.plot()

    @property
    def pids(self) -> List[int]:
        rv = []
        for m in self._markers:
            rv.extend( m.pids )
        return rv

    @property
    def tpids(self) -> List[int]:
        rv = []
        for m in self._markers:
            if m.cid == 0: rv.extend( m.pids )
        return rv

    @log_timing
    def plot(self):
        self._overlay = hv.NdOverlay(self._curves)
        self._plot = datashade(self._overlay, normalization='linear', aggregator=ds.count()).opts(self.opts)
        lgm().log( f"dsGraphPlot->plot: {self.plot.__class__}")



