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
        self.init_plot()
        self._enabled = False
        self.axis_to_data = ax.transAxes + ax.transData.inverted()
        self.data_to_axis = self.axis_to_data.inverted()

    def set_enabled(self, enable: bool ):
        self._enabled = enable
        lgm().log(f"PointsInteractor.set_enabled: {enable}")
        if enable:  self.enable_callbacks()
        else:       self.disable_callbacks()

    def init_plot(self):
        self.points: PathCollection = self.ax.scatter([], [], s=50, zorder=3, alpha=1.0 )
        self.points.set_edgecolor([0, 0, 0])
        self.points.set_linewidth(2)
        self._cidkey, self._cidmouse = -1, -1
        self.plot()

    def enable_callbacks(self):
        self._cidkey = self.canvas.mpl_connect( 'key_press_event', self.on_key_press )
        self._cidmouse = self.canvas.mpl_connect('button_press_event', self.on_button_press)

    def disable_callbacks(self):
        for cid in [ self._cidkey, self._cidmouse ]:
            self.canvas.mpl_disconnect( cid )

    def get_points( self ) -> Tuple[ List[float], List[float], List[str] ]:
        lgm().log( f"Attempt to call unimplemented method PointsInteractor.get_markers")
        return [], [], []

    def on_key_press(self, event: KeyEvent ):
        lgm().log( f"Attempt to call unimplemented method PointsInteractor.on_key_press")

    def on_button_press(self, event: MouseEvent):
        lgm().log( f"Attempt to call unimplemented method PointsInteractor.on_button_press")

    @exception_handled
    def plot( self ):
        ycoords, xcoords, colors = self.get_points()
        lgm().log(f" ** plot point_selection image, nmarkers = {len(ycoords)}")
        if len(ycoords) > 0:
            self.points.set_offsets(np.c_[xcoords, ycoords])
            self.points.set_facecolor(colors)
        else:
            offsets = np.ma.column_stack([[], []])
            self.points.set_offsets(offsets)
        self.canvas.draw_idle()

    def clear( self ):
        offsets = np.ma.column_stack([[], []])
        self.points.set_offsets( offsets )
        self.plot()

    def toggleVisible(self):
        new_alpha = 1.0 if (self.points.get_alpha() == 0.0) else 0.0
        self.points.set_alpha( new_alpha )
        self.canvas.draw_idle()