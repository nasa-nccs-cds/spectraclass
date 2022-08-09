from collections import OrderedDict
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from typing import List, Union, Tuple, Optional, Dict, Callable
from matplotlib.backend_bases import PickEvent, MouseEvent, KeyEvent,  MouseButton  # , NavigationToolbar2
import numpy as np
from matplotlib.collections import PathCollection

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

    def set_alpha(self, alpha: float ):
        self.points.set_alpha( alpha )

    def init_plot(self):
        self.points: PathCollection = self.ax.scatter([], [], s=50, zorder=5, alpha=1.0 )
        self.points.set_edgecolor([0, 0, 0])
        self.points.set_linewidth(2)
        self.highlights: PathCollection = self.ax.scatter([], [], s=100, zorder=100, alpha=0.75, marker="X", color="white" )
        self.highlights.set_linewidth(1)
        self.highlights.set_edgecolor( "black" )
        self.probes: PathCollection = self.ax.scatter([], [], s=50, zorder=5, alpha=1.0, color="white" )
        self.probes.set_edgecolor([0, 0, 0])
        self.probes.set_linewidth(2)
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

    def on_button_press(self, event: MouseEvent):
        lgm().log( f"Attempt to call unimplemented method PointsInteractor.on_button_press")

    @log_timing
    def plot( self, **kwargs ):
        lgm().log(f" MAP Marker plot" )
        ycoords, xcoords, colors = self.get_points( probes=False )
        if len(ycoords) > 0:
            lgm().log(f" --> MAP Markers: {len(ycoords)}")
            self.points.set_offsets(np.c_[xcoords, ycoords])
            self.points.set_facecolor(colors)
        else:
            offsets = np.ma.column_stack([[], []])
            self.points.set_offsets(offsets)

        ycoords, xcoords, _ = self.get_points( probes=True )
        if len(ycoords) > 0:
            lgm().log(f" --> MAP Probes: {len(ycoords)}")
            self.probes.set_offsets(np.c_[xcoords, ycoords])
        else:
            offsets = np.ma.column_stack([[], []])
            self.probes.set_offsets(offsets)

        if kwargs.get('clear_highlights', False): self._highlight_points = []
        ycoords, xcoords, cids = self.get_highlight_points()
        if len(ycoords) > 0:
            lgm().log(f" --> MAP Hightlghting {len(ycoords)} points: {tuple(self._highlight_points)} ")
            self.highlights.set_offsets(np.c_[xcoords, ycoords])
        else:
            offsets = np.ma.column_stack([[], []])
            self.highlights.set_offsets(offsets)
        self.canvas.draw_idle()

    def clear( self ):
        offsets = np.ma.column_stack([[], []])
        self.points.set_offsets( offsets )
        self.highlights.set_offsets(offsets)
        self.probes.set_offsets(offsets)
        self.plot()

    def toggleVisible(self):
        new_alpha = 1.0 if (self.points.get_alpha() == 0.0) else 0.0
        self.points.set_alpha( new_alpha )
        self.highlights.set_alpha( new_alpha )
        self.canvas.draw_idle()