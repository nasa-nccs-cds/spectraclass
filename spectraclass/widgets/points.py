from collections import OrderedDict
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from typing import List, Union, Tuple, Optional, Dict, Callable
import numpy as np

class PointsInteractor:

    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.ax = ax
        self._highlight_points: List[Tuple[int,int]] = []
        self.init_plot()
        self._enabled = False
        self.axis_to_data = ax.transAxes + ax.transData.inverted()
        self.data_to_axis = self.axis_to_data.inverted()
        self.canvas.mpl_connect('button_press_event', self.on_button_press)

    def set_enabled(self, enable: bool ):
        self._enabled = enable
        lgm().log(f"PointsInteractor.set_enabled: {enable}")

    def init_plot(self):
        self._cidkey, self._cidmouse = -1, -1
        self.plot()

    def highlight_points(self, pids: List[int], cids: List[int] ):
        self._highlight_points = zip(pids, cids)
        self.plot()

    def clear_highlights(self ):
        self.plot( clear_highlights = True )

    def get_points( self, probes: bool = False ) -> Tuple[ List[float], List[float], List[str] ]:
        lgm().log( f"Attempt to call unimplemented method PointsInteractor.get_points")
        return [], [], []

    def get_highlight_points( self ) -> Tuple[ List[float], List[float], List[int] ]:
        lgm().log( f"Attempt to call unimplemented method PointsInteractor.get_highlight_points")
        return [], [], []

    def on_button_press( self, event ):
        lgm().log( f"Attempt to call unimplemented method PointsInteractor.on_button_press")

    @log_timing
    def plot( self, **kwargs ):
        lgm().log(f" MAP Marker plot" )
        ycoords, xcoords, colors = self.get_points( probes=False )
        if len(ycoords) > 0:
            lgm().log(f" --> MAP Markers: {len(ycoords)}")
        else:
            offsets = np.ma.column_stack([[], []])

        ycoords, xcoords, _ = self.get_points( probes=True )
        if len(ycoords) > 0:
            lgm().log(f" --> MAP Probes: {len(ycoords)}")
        else:
            offsets = np.ma.column_stack([[], []])

        if kwargs.get('clear_highlights', False): self._highlight_points = []
        ycoords, xcoords, cids = self.get_highlight_points()
        if len(ycoords) > 0:
            lgm().log(f" --> MAP Hightlghting {len(ycoords)} points: {tuple(self._highlight_points)} ")
        else:
            offsets = np.ma.column_stack([[], []])
        self.canvas.draw_idle()

    def clear( self ):
        offsets = np.ma.column_stack([[], []])
        self.plot()

    def toggleVisible(self):
        self.canvas.draw_idle()