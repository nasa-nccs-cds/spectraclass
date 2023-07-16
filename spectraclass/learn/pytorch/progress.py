from typing import List, Union, Tuple, Optional, Dict, Type, Callable
import torch, time, os
from spectraclass.gui.control import ufm
from holoviews.streams import Stream, param
from panel.layout.base import Panel
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from torch import Tensor, argmax
import xarray as xa, numpy as np
import holoviews as hv, panel as pn
import hvplot.xarray  # noqa

Loss = Stream.define( 'Loss', loss=0.0 )

class ProgressPanel(param.Parameterized):
    loss = param.List( default=[], doc="Loss values")

    def __init__(self, niter: int, abort_callback: Callable, **kwargs ):
        param.Parameterized.__init__( self, **kwargs )
        self.niter = niter
        self._progress = pn.indicators.Progress( name='Iterations', value=0, width=200, max=niter )
        self._log = pn.pane.Markdown("Iteration: 0", width=150)
        self._losses = []
        self._abort = pn.widgets.Button( name='Abort', button_type='warning', width=100 )
        self._abort.on_click( abort_callback )
        self.loss_stream: Stream = Loss( loss=0.0 )
        self._loss_plot = hv.DynamicMap( self.plot_losses, streams=[ self.loss_stream ] )

    @exception_handled
    def update(self, iteration: int, message: str, loss: float ):
        self._progress.value = iteration
        self._log.object = message
        lgm().log( f"UPDATE: iteration={iteration}, message={message}, loss={loss}")
        self.loss_stream.event( loss=loss )

    @exception_handled
    def plot_losses(self, loss: float = 0.0 ):
        first_element = (len(self._losses) == 1) and (self._losses[0] == 0.0)
        if first_element:   self._losses[0] = loss
        else:               self._losses.append(loss)
        iterations: np.ndarray = np.arange( len(self._losses) )
        lgm().log( f"Plot Losses: {len(self._losses)} values")
        loss_table: hv.Table = hv.Table( (iterations, np.array(self._losses) ), 'Iteration', 'Loss' )
        return hv.Curve(loss_table).opts(width=500, height=250, ylim=(0,1.0), xlim=(0,self.niter))  #  line_width=1, line_color="black",

    def panel(self) -> pn.WidgetBox:
        progress = pn.Row( self._progress, self._log, self._abort )
        return pn.WidgetBox( "### Progress", progress, self._loss_plot )