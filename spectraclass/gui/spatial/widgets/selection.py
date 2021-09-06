from typing import List, Dict, Tuple, Optional
import math, atexit, os, traceback
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.widgets import PolygonSelector,  RectangleSelector, _SelectorWidget
from matplotlib.lines import Line2D
from spectraclass.util.logs import LogManager, lgm
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.axes import Axes
from spectraclass.gui.control import UserFeedbackManager, ufm
from matplotlib.patches import Rectangle

class PointsSelector(_SelectorWidget):

    def __init__(self, ax, onselect=None, useblit=True, lineprops=None,  button=None):
        super().__init__(ax, onselect, useblit=useblit, button=button)
        self.initMarkers()

    def clearMarkers( self ):
        self.set_alpha( 0.0 )
        self.initMarkers()

    def set_alpha(self, new_alpha ):
        self.marker_plot.set_alpha( new_alpha )
        self.update()

    @property
    def verts(self) -> List[Tuple]:
        return list(zip( self.xcoords, self.ycoords ))

    def initMarkers(self):
        self.ycoords = []
        self.xcoords = []
        self.marker_plot: PathCollection = self.ax.scatter([], [], s=50, zorder=3, alpha=1, picker=True)
        self.marker_plot.set_edgecolor([0, 0, 0])
        self.marker_plot.set_linewidth(2)
#        self.figure.canvas.mpl_connect('pick_event', self.mpl_pick_marker)
        self.update()

    def plotMarkers( self ):
        if len(self.ycoords) > 0:
            lgm().log( f"plotMarkers: {self.xcoords} {self.ycoords}")
            self.marker_plot.set_offsets(np.c_[self.xcoords, self.ycoords])
            self.marker_plot.set_facecolor( "white" )
        else:
            offsets = np.ma.column_stack([[], []])
            self.marker_plot.set_offsets(offsets)
        self.update()

    def onpress(self, event):
        self.press(event)

    def _press(self, event):
        mouse_coords = self._get_data(event)
        self.xcoords.append( mouse_coords[0] )
        self.ycoords.append( mouse_coords[1] )

    def _release(self, event):
        self.plotMarkers()


class LassoSelector(_SelectorWidget):

    def __init__(self, ax, onselect=None, useblit=True, lineprops=None,  button=None):
        super().__init__(ax, onselect, useblit=useblit, button=button)
        self.verts = None
        if lineprops is None:
            lineprops = dict()
        lineprops.update(animated=self.useblit, visible=False)
        self.line = Line2D([], [], **lineprops)
        self.ax.add_line(self.line)
        self.artists = [self.line]

    def onpress(self, event):
        self.press(event)

    def _press(self, event):
        self.verts = [self._get_data(event)]
        self.line.set_visible(True)

    def onrelease(self, event):
        self.release(event)

    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.onselect(self.verts)

    def _onmove(self, event):
        if self.verts is None:
            return
        self.verts.append(self._get_data(event))
        self.line.set_data(list(zip(*self.verts)))
        self.update()

class SelectionTool(ToolToggleBase):

    def __init__( self, figure, **kwargs ):
        super().__init__( figure.canvas.manager.toolmanager, self.__class__.__name__, toggled=False, **kwargs )
        self.figure = figure

    def selection(self) -> List[Tuple]:
        raise NotImplementedError()

    def enable(self, *args ):
        raise NotImplementedError()

    def disable(self, *args):
        self.disconnect()
        super().disable(self)

    def onselect(self, *args):
        print(f"onselect: {args} ")
        self.canvas.draw_idle()

    def disconnect(self):
        raise NotImplementedError()

class PointSelectionTool(SelectionTool):
    default_keymap = 'p'
    description = 'Select points'

    def __init__( self, figure, **kwargs ):
        super().__init__( figure, **kwargs )
        self.markers: PointsSelector = None

    def selection(self) -> List[Tuple]:
        return self.markers.verts

    def enable(self, *args ):
       ufm().show( "Enable Point Selection")
       self.markers = PointsSelector(self.figure.axes[0], self.onselect, useblit=False, button=[1] )

    def disconnect(self):
        print(f"DISCONNECT")
        self.markers.clearMarkers()
        self.markers = None
        self.canvas.draw_idle()

class LassoSelectionTool(SelectionTool):
    default_keymap = 'r'
    description = 'Select region with curve'

    def __init__( self, figure, **kwargs ):
        super().__init__( figure, **kwargs )
        self.curve: LassoSelector = None

    def selection(self) -> List[Tuple]:
        return self.curve.verts

    def enable(self, *args ):
       ufm().show( "Enable Lasso Selection")
       self.curve: LassoSelector = LassoSelector(self.figure.axes[0], self.onselect, useblit=False, button=[1] )

    def disconnect(self):
        print(f"DISCONNECT")
        self.curve.set_visible(False)
        self.curve.set_active(False)
        self.curve = None
        self.canvas.draw_idle()

class RectangleSelectionTool(SelectionTool):
    default_keymap = 'r'
    description = 'Select Rectangular Region'

    def __init__( self, figure, **kwargs ):
        super().__init__( figure, **kwargs )
        self.poly: RectangleSelector = None

    def selection(self) -> List[Tuple]:
        xc, yc = self.poly.corners
        return list( zip( xc, yc ) )

    def enable(self, *args ):
       ufm().show( "Enable Rectangle Selection")
       self.poly = RectangleSelector( self.figure.axes[0], self.onselect, drawtype='box', useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    def disconnect(self):
        print(f"DISCONNECT")
        self.poly.set_visible(False)
        self.poly.set_active( False )
        self.poly = None
        self.canvas.draw_idle()

class PolygonSelectionTool(SelectionTool):
    default_keymap = 'p'
    description = 'Select Polygonal Region'

    def __init__( self, figure, **kwargs ):
        super().__init__( figure, **kwargs )
        self.figure = figure
        self.poly = None

    def selection(self) -> List[Tuple]:
        assert ( (self.poly is not None) and self.poly._polygon_completed ), "Region boundary is not well defined"
        return self.poly.verts

    def enable(self, *args ):
       ufm().show( "Enable Polygon Selection")
       self.poly = PolygonSelector( self.figure.axes[0], self.onselect )

    def disconnect(self):
        print(f"DISCONNECT")
        self.poly.set_visible(False)
        self.poly.disconnect_events()
        self.poly = None
        self.canvas.draw_idle()