import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.patches import Polygon
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from shapely.geometry import Polygon as SPolygon
from matplotlib.backend_bases import MouseEvent, KeyEvent
from typing import List, Union, Tuple, Optional, Dict, Callable

def dist(x, y):
    d = x - y
    return np.sqrt(np.dot(d, d))

class PolyRec:
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, polyId, ax, x, y, c="grey", on_change: Callable = None ):
        self.ax = ax
        self.color = c
        self.canvas = ax.figure.canvas
        self.polyId = polyId
        self.selected = False
        xs, ys = np.array( [x,x] ), np.array( [y,y] )
        self.poly = Polygon( np.column_stack([xs,ys]), facecolor=self.color, closed=False )
        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r' )
        if on_change: self.cid = self.poly.add_callback( on_change )
        else: self.cid = None
        self.poly.set_zorder(25)
        ax.add_patch( self.poly )
        ax.add_line( self.line )
        self.indx = -1

    def set_alpha(self, alpha: float ):
        self.poly.set_alpha( alpha )

    def to_shapely(self) -> SPolygon:
        return SPolygon( self.poly.get_xy() )

    def contains_point(self, event: MouseEvent ) -> bool:
        return self.poly.contains_point( (event.x,event.y) )

    def vertex_selected( self, event: MouseEvent ):
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        d0 = d[ indseq[0] ]
        selected = (d0 < self.epsilon)
        self.indx = indseq[0] if selected else -1
        return ( self.indx > -1 )

    def clear_vertex_selection(self):
        self.indx = -1

    def _update(self):
        self.line.set_data(zip(*self.poly.xy))

    def insert_point(self, event ):
        x, y = event.xdata, event.ydata
        self.poly.xy = np.row_stack( [ self.poly.xy, np.array( [x, y] ) ] )
        self.draw()

    def complete( self ):
        self.poly.xy[-1] = self.poly.xy[0]
        self.line.set_visible(False)
        self.poly.set_closed(True)
        self.ax.draw_artist(self.line)

    def update(self):
        self.line.set_data(zip(*self.poly.xy))
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)

    def draw(self):
        self.canvas.draw_idle()

    def set_selected(self, selected: bool ):
        self.selected = selected
        self.line.set_visible(selected)
        self.ax.draw_artist(self.line)

    def drag_vertex(self, event ):
        x, y = event.xdata, event.ydata
        self.poly.xy[ self.indx ] = x, y
        indx1 = self.poly.xy.shape[0]-1
        if self.indx == 0:     self.poly.xy[indx1] = self.poly.xy[0]
        if self.indx == indx1: self.poly.xy[0]     = self.poly.xy[indx1]
