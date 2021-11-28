import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseEvent
import logging, os
from typing import List, Union, Tuple, Optional, Dict, Callable

log_file = os.path.expanduser('~/.spectraclass/logging/geospatial.log')
file_handler = logging.FileHandler(filename=log_file, mode='w')
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

class PolyMode:
    NONE = 0
    CREATING = 1
    EDITING = 2

def dist(x, y):
    """
    Return the distance between two points.
    """
    d = x - y
    return np.sqrt(np.dot(d, d))

def dist_point_to_segment(p, s0, s1):
    """
    Get the distance of a point to a segment.
      *p*, *s0*, *s1* are *xy* sequences
    This algorithm from
    http://www.geomalgorithms.com/algorithms.html
    """
    v = s1 - s0
    w = p - s0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return dist(p, s0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return dist(p, s1)
    b = c1 / c2
    pb = s0 + b * v
    return dist(p, pb)

class PolyRec:

    def __init__(self, pid, ax,  x, y, c, on_change: Callable = None ):
        self.ax = ax
        self.color = c
        self.canvas = ax.figure.canvas
        self.pid = pid
        xs, ys = np.array( [x,x] ), np.array( [y,y] )
        self.poly = Polygon( np.column_stack([xs,ys]), animated=True, facecolor=self.color )
        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
        if on_change: self.cid = self.poly.add_callback( on_change )
        else: self.cid = None
        ax.add_patch(self.poly)
        ax.add_line(self.line)
        self.indx = None

    def contains_point(self, event: MouseEvent ) -> bool:
        return self.poly.contains_point( (event.x,event.y) )

    def _update(self):
        self.line.set_data(zip(*self.poly.xy))

    def insert_point(self, x, y ):
        self.poly.xy = np.insert( self.poly.xy, len(self.poly.xy), [x, y], axis=0 )
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
        self.line.set_visible(selected)
        self.ax.draw_artist(self.line)

class PolygonInteractor:

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax):
        self.ax = ax
        self.polys: List[PolyRec] = []
        self.prec: PolyRec = None
        self.mode = PolyMode.NONE
        self.fill_color = "grey"

        canvas = ax.figure.canvas
        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('key_release_event', self.on_key_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

    def add_poly( self, x, y ):
        pid = len(self.polys)
        self.prec = PolyRec( pid, ax, x, y, self.fill_color, self.poly_changed )
        self.polys.append( self.prec )
        return self.prec

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for prec in self.polys:
            self.ax.draw_artist(prec.poly)
            self.ax.draw_artist(prec.line)

    def poly_changed(self, poly):
        if self.prec is not None:
            vis = self.prec.line.get_visible()
            Artist.update_from(self.prec.line, poly)
            self.prec.line.set_visible(vis)

    def get_selected_poly( self, event ) -> Optional[PolyRec]:
        for prec in self.polys:
            if prec.contains_point( event ):
                return prec
        return None

    def get_ind_under_point( self, event ) -> Optional[PolyRec]:
        for prec in self.polys:
            xy = np.asarray(prec.poly.xy)
            xyt = prec.poly.get_transform().transform(xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.hypot(xt - event.x, yt - event.y)
            indseq, = np.nonzero(d == d.min())
            ind = indseq[0]
            logger.info( f" get_ind_under_point-> {prec.pid}:{prec.indx}, d={d} xt=[{xt},{yt}] et=[{event.x},{event.y}], ind={ind}")
            if d[ind] < self.epsilon:
                prec.indx = ind
                return prec
        return None

    def close_poly(self):
        self.prec.complete()
        self.prec = None
        self.draw()

    def on_button_press(self, event: MouseEvent ):
        if event.inaxes is None: return

        if event.button == 1:
            if self.mode == PolyMode.CREATING:
                if self.prec is None:
                    self.add_poly(event.xdata, event.ydata)
                else:
                    self.prec.insert_point( event.xdata, event.ydata )
                    if event.dblclick:
                        self.close_poly()
            else:
                selection = self.get_selected_poly( event )
                if selection is not None:
                    for prec in self.polys:
                        prec.set_selected( prec.pid == selection.pid )
                    self.draw()
        elif event.button == 3:
            pass

    def on_button_release(self, event):
        if event.button != 1: return
        self._ind = None

    def on_key_release(self, event):
        if (self.mode == PolyMode.CREATING) and (self.prec != None):
            self.prec.complete()
            self.prec = None
            self.draw()
        self.mode = None

    def on_key_press(self, event):
        if not event.inaxes:
            return
        if   event.key == 'c':  self.mode = PolyMode.CREATING
        elif event.key == 'e':  self.mode = PolyMode.EDITING
        elif event.key == 'r':  self.fill_color = "red"
        elif event.key == 'g':  self.fill_color = "green"
        elif event.key == 'b':  self.fill_color = "blue"
        elif event.key == 'p':  self.fill_color = "purple"


        #     self.showverts = not self.showverts
        #     self.line.set_visible(self.showverts)
        #     if not self.showverts:
        #         self._ind = None
        # elif event.key == 'd':
        #     ind = self.get_ind_under_point(event)
        #     if ind is not None:
        #         self.poly.xy = np.delete(self.poly.xy,
        #                                  ind, axis=0)
        #         self.line.set_data(zip(*self.poly.xy))
        # elif event.key == 'i':
        #     xys = self.poly.get_transform().transform(self.poly.xy)
        #     p = event.x, event.y  # display coords
        #     for i in range(len(xys) - 1):
        #         s0 = xys[i]
        #         s1 = xys[i + 1]
        #         d = dist_point_to_segment(p, s0, s1)
        #         if d <= self.epsilon:
        #             self.poly.xy = np.insert(
        #                 self.poly.xy, i+1,
        #                 [event.xdata, event.ydata],
        #                 axis=0)
        #             self.line.set_data(zip(*self.poly.xy))
        #             break
        # if self.line.stale:
        #     self.canvas.draw_idle()

    def on_mouse_move(self, event):
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if self.prec is not None:
            if self.mode == PolyMode.CREATING:
                x, y = event.xdata, event.ydata
                self.prec.poly.xy[-1] = x, y
                self.draw()
            elif self.mode == PolyMode.EDITING:
                pass

    def draw(self):
        self.canvas.restore_region(self.background)
        for prec in self.polys: prec.update()
        self.canvas.blit(self.ax.bbox)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots()
    p = PolygonInteractor(ax)
    ax.set_title('Patch selection test')
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    plt.show()