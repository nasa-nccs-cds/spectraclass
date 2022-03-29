import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from spectraclass.widgets.polygon import PolyRec
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from matplotlib.backend_bases import MouseEvent, KeyEvent
from typing import List, Union, Tuple, Optional, Dict, Callable

def dist(x, y):
    d = x - y
    return np.sqrt(np.dot(d, d))

class PolygonInteractor:

    def __init__(self, ax):
        self.ax = ax
        self.polys: List[PolyRec] = []
        self.prec: PolyRec = None
        self.enabled = False
        self.editing = False
        self.creating = False
        self._fill_color = "grey"
        self._cid = 0
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move )
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.poly_index = 0
        self.cids = []

    def set_alpha(self, alpha: float ):
        for poly in self.polys:
            poly.set_alpha( alpha )

    @exception_handled
    def set_enabled(self, enabled: bool ):
        if enabled != self.enabled:
            self.enabled = enabled

    def set_class(self, cid: int ):
        from spectraclass.model.labels import LabelsManager, lm
        self._cid = cid
        self._fill_color = lm().colors[ self._cid ]

    def add_poly( self, event ):
        if not self.in_poly(event):
            x, y = event.xdata, event.ydata
            self.poly_index = self.poly_index + 1
            self.prec = PolyRec( self.poly_index, self.ax, x, y, self._fill_color, self.poly_changed )
            self.polys.append( self.prec )
            self.creating = True
        return self.prec

    @exception_handled
    def on_draw(self, event):
 #       lgm().log( "POLY->on_draw")  # lgm().log( f"POLY->close: points = {self.prec.poly.get_xy().tolist()}")
        for prec in self.polys:
            self.ax.draw_artist(prec.poly)
            self.ax.draw_artist(prec.line)

    def poly_changed(self, poly):
        if self.prec is not None:
            vis = self.prec.line.get_visible()
            Artist.update_from(self.prec.line, poly)
            self.prec.line.set_visible(vis)

    def in_poly( self, event ) -> Optional[PolyRec]:
        for prec in self.polys:
            if prec.contains_point( event ):
                return prec
        return None

    def select_poly(self, event):
        self.prec = self.in_poly( event )
        selected_pid = self.prec.polyId if (self.prec is not None) else -1
        for prec in self.polys:
            prec.set_selected(prec.polyId == selected_pid)
        self.draw()

    def delete_selection(self):
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.gui.points3js import PointCloudManager, pcm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if self.prec is not None:
            self.polys.remove( self.prec )
            self.prec.poly.remove()
            self.prec.line.remove()
            marker = tm().get_region_marker( self.prec )
            if marker is not None:
                gpm().remove_marker(marker)
                pcm().deleteMarkers( marker.pids )
            self.prec = None
            self.canvas.draw_idle()

    def close_poly(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.application.controller import app
        self.prec.complete()
        self.creating = False
        lgm().log( f"POLY->close: points = {self.prec.poly.get_xy().tolist()}")
        self.draw()
        marker = tm().get_region_marker( self.prec )
        app().add_marker( "map", marker )
        self.prec = None

    @exception_handled
    def on_button_press(self, event: MouseEvent ):
        if self.enabled and (event.inaxes is not None):
            x, y = event.xdata, event.ydata
            lgm().log( f"POLYINTER: button {event.button} press: enabled={self.enabled}, creating={self.creating}, point={[x,y]}")
            lgm().log(f"  ***-> CALLBACKS: {self.canvas.callbacks.callbacks['motion_notify_event']}")
            if event.button == 1:
                if self.enabled:
                    if self.prec is None:
                        self.add_poly( event )
                    else:
                        if self.creating:
                            self.prec.insert_point( event )
                        else:
                            self.editing = self.prec.vertex_selected( event )
            elif event.button == 3:
                if self.creating:
                    self.prec.insert_point( event )
                    self.close_poly()
                else:
                    self.select_poly( event )

    @exception_handled
    def on_button_release(self, event):
        if self.enabled and (event.inaxes is not None):
            if self.prec is not None:
                self.prec.clear_vertex_selection()
                self.editing = False

    @exception_handled
    def on_key_press(self, event: KeyEvent ):
        if self.enabled and (event.inaxes is not None):
            if event.key == 'backspace':  self.delete_selection()

    @exception_handled
    def on_mouse_move(self, event):
#        lgm().log(f"POLYEDIT-> MOUSE-MOVE: {self.enabled} {event.inaxes is not None} {self.editing} {self.creating} {[event.xdata, event.ydata]}")
        if self.enabled and (event.inaxes is not None):
            if (self.editing or self.creating):
                self.prec.drag_vertex( event )
                self.draw()

    @exception_handled
    def draw(self):
        for prec in self.polys: prec.update()
        self.canvas.draw_idle()

    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Polygon
    #
    # fig, ax = plt.subplots()
    # p = PolygonInteractor(ax)
    # ax.set_title('Patch selection test')
    # ax.set_xlim((-2, 2))
    # ax.set_ylim((-2, 2))
    # plt.show()

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