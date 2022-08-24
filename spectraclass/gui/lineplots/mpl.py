from matplotlib.backend_bases import PickEvent, MouseEvent, MouseButton, KeyEvent  # , NavigationToolbar2
from matplotlib.lines import Line2D
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
from collections import OrderedDict
import time, xarray as xa
import numpy as np
from spectraclass.data.base import DataManager, dm
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import matplotlib.pyplot as plt
from .manager import LinePlot
import ipywidgets as ipw

class LineRec:

    def __init__(self, line: Optional[Line2D], pid: int, cid: int, **kwargs ):
        self.line: Optional[Line2D] = line
        self.pid: int = pid
        self.cid: int = cid
        self.mpids: List[int] = kwargs.get( 'mpids', [] )

    def clear(self):
        if self.line is not None:
            self.line.remove()
            self.line = None

    @property
    def id(self) -> int:
        return -1 if self.line is None else self.lid(self.line)

    @classmethod
    def lid( cls, line: Line2D ) -> int:
        return id(line) # line.get_ydata().mean()

class mplGraphPlot(LinePlot):

    def __init__( self, index: int, **kwargs ):
        LinePlot.__init__( self, index, **kwargs )
        self.rlines: List[Line2D] = []
        self._max_graph_group_size = 100
        self.ax: plt.Axes = None
        self.fig: plt.Figure = None
        self.selected_pid: int = -1
        self.lrecs: OrderedDict[int, LineRec] = OrderedDict()
        self.init_figure( **kwargs )

    @property
    def ids(self) -> List[float]:
        return [lrec.id for lrec in self.lrecs.values()]

    def get_lrec( self, line: Line2D ) -> Optional[LineRec]:
        lid = LineRec.lid( line )
        for lrec in self.lrecs.values():
            if lrec.id == lid: return lrec
        return None

    def get_selected_lrec( self ) -> Optional[LineRec]:
        if self.selected_pid == -1: return None
        return self.lrecs[ self.selected_pid ]

    @property
    def pids(self) -> List[int]:
        return list( self.lrecs.keys() )

    @property
    def tpids(self) -> List[int]:
        return [ pid for (pid,lrec) in self.lrecs.items() if (lrec.cid == 0) ]

    def init_figure(self, **kwargs):
        if self.fig is None:
            if not self.standalone: plt.ioff()
            self.fig: plt.Figure = plt.figure( self.index, figsize = (6, 4) )
            if len(self.fig.axes) == 0: self.fig.add_subplot(111)
            self.ax = self.fig.axes[0]
            self.ax.grid(True)
            self.ax.set_autoscaley_on(True)
            self.ax.set_title(f'Point Spectra {self.index}', fontsize=12)
            self.fig.canvas.mpl_connect('pick_event', self.onpick )
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            if not self.standalone: plt.ion()

    def gui(self):
        return ipw.VBox([ self.fig.canvas, self.control_panel() ] )

    def control_panel(self) -> ipw.DOMWidget:
        mark_button = ipw.Button(description="Reclassify", layout=ipw.Layout(width='120px'), border='1px solid dimgrey')
        mark_button.on_click( self.mark_selection )
        unmark_button = ipw.Button(description="Unclassify", layout=ipw.Layout(width='120px'), border='1px solid dimgrey')
        unmark_button.on_click( self.delete_selection )
        return ipw.HBox([ mark_button, unmark_button ] )

    def clear(self, reset: bool = True ):
        for lrec in self.lrecs.values(): lrec.clear()
        for rline in self.rlines: rline.remove()
        self.rlines = []
        if reset: self.lrecs = OrderedDict()

    def clearTransients(self):
        new_lrecs = {}
        for (pid, lrec) in self.lrecs.items():
            if lrec.cid == 0: lrec.clear()
            else: new_lrecs[pid] = lrec
        for rline in self.rlines:
            rline.remove()
        self.rlines = []
        self.lrecs = new_lrecs

    @log_timing
    @exception_handled
    def addMarker( self, m: Marker ):
        lgm().log(f"mplGraphPlot: Add Marker[{m.size}]: cid={m.cid}, pids[:10]={m.pids[:10]}")
        if m.size > 0:
            self.clearTransients()
            if len(m.pids) == 1:    self.plot_line( m.pids[0], m.cid )
            else:                   self.plot_lines( m.pids.tolist(), m.cid )

    @log_timing
    def plot_line(self, pid: int, cid: int ):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        selected: bool = (self.selected_pid >= 0)
        selection = (pid == self.selected_pid)
        alpha = (1.0 if selection else 0.2) if selected else 1.0
        color = lm().graph_colors[cid]
        lw = 2.0 if selection else 1.0
        lrec = LineRec(None, pid, cid)
        self.lrecs[pid] = lrec
        x,y = self.lx(lrec.pid), self.ly(lrec.pid)
        if y is not None:
            lines = self.ax.plot( x,y, picker=True, pickradius=2, color=color, alpha=alpha, linewidth=lw )
            lrec.line = lines[0]
            if (not self._use_model) and (self.ry.size > 0):
                self.rlines.extend( self.ax.plot( x, self.lry(lrec.pid), color="grey" ) )
            self.ax.figure.canvas.draw_idle()
        else:
            ufm().show(f"Points out of bounds","red")


    @log_timing
    def plot_lines(self, mpids: List[int], cid: int ):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        color = lm().graph_colors[cid]
        skip_index = max( len(mpids)//self._max_graph_group_size, 1 )
        pids = mpids[::skip_index]
        x,y = self.lx(pids), self.ly(pids)
        if y is not None:
            lrecs = [ LineRec(None, pid, cid, mpids=mpids) for pid in pids ]
            for lrec in lrecs: self.lrecs[lrec.pid] = lrec
            lines = self.ax.plot( x, y, picker=True, pickradius=2, color=color, alpha=0.2, linewidth=1.0 )
            self.ax.figure.canvas.draw_idle()
            for (lrec, line) in zip(lrecs, lines): lrec.line = line
        else:
            ufm().show(f"Points out of bounds","red")

    def expose_nearby_lines(self, pid: int ):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        y: np.ndarray = self.y
        sely: np.ndarray = self.ly(pid)
        ufm().show(f"Compare: y: {y.shape}, self: {sely.shape}", "blue")

    @exception_handled
    def get_plotspecs(self):
        from spectraclass.model.labels import LabelsManager, lm
        colors, alphas, lws = [], [], []
        test_ids = {}
     #   lgm().log( f"create plotspecs for {len(self.lrecs.items())} lines, pids({self.selected_pid})->{self.pids}")
        for (pid, lrec) in self.lrecs.items():
            selection = ( pid == self.selected_pid )
            alphas.append( 1.0 if selection else 0.2 )
            colors.append( lm().graph_colors[ lrec.cid ] )
            lws.append( 2.0 if selection else 1.0 )
            idx = len(colors)-1
            if ( pid in [ 22025, 22663 ] ) or selection:
                test_ids[ idx ] = pid
        if self.selected_pid >= 0:
            lgm().log(f" ^^^ get_plotspecs-test: " )
            for tpid in [22025, 22663 ]:
                lrec = self.lrecs[tpid]
                lgm().log(f" ----->>> pid={tpid}, cid={lrec.cid} ")
        return dict( color=colors, alpha=alphas, lw=lws, test_ids = test_ids )

    @property
    def cids( self ):
        return  [ lrec.cid for lrec in self.lrecs.values() ]

    @log_timing
    def plot( self, clear_selection = False ):
        self.ax.title.text = self.title
        if clear_selection: self.selected_pid = -1
        ps = self.get_plotspecs()
        colors, alphas, linewidths = ps['color'], ps['alpha'], ps['lw']
        test_colors = { pid: colors[idx] for (idx,pid) in ps['test_ids'].items() }
        lgm().log(f"GRAPHPlot->test colors: {test_colors}")
        try:
            self.ax.set_prop_cycle( color=colors, alpha=alphas, linewidth=linewidths )
        except Exception as err:
            lgm().log(f"set_prop_cycle: color={colors}, alpha=alphas, linewidth={linewidths}")
            lgm().log( f"## Error setting property cycle: {err}")
        self.update_graph()

    @log_timing
    def update_graph(self, **kwargs ):
        self.clear( False )
        lines: List[Line2D] = self.ax.plot( self.x, self.y, picker=True, pickradius=2, **kwargs )
        if (not self._use_model) and (self.ry.size > 0):
            self.rlines: List[Line2D] = self.ax.plot( self.x, self.ry, color="grey", **kwargs )
        for (line, lrec) in zip(lines, self.lrecs.values()): lrec.line = line
        self.fig.canvas.draw()

    @exception_handled
    def onpick(self, event: PickEvent ):
        from spectraclass.gui.spatial.map import MapManager, mm
        line: Line2D = event.artist
        selected_lrec = self.get_lrec( line )
        if selected_lrec is None:
            lgm().log( f"\nonpick: line={LineRec.lid(line)}, lines={self.ids}, inlist={LineRec.lid(line) in self.ids}")
        else:
            self.selected_pid = selected_lrec.pid
            mm().highlight_points( [self.selected_pid], [selected_lrec.cid] )
            self.plot()

    def mark_selection( self, *args ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.model.labels import LabelsManager, lm
        if self.selected_pid >= 0:
            lrec: LineRec = self.get_selected_lrec()
            lrec.cid = lm().current_cid
            mm().mark_point( lrec.pid, cid=lrec.cid )
            self.expose_nearby_lines( lrec.pid )
            mm().plot_markers_image( clear_highlights=True )
            lgm().log( f"MARK SELECTION: cid={lrec.cid}, pid={lrec.pid}")
            self.plot()

    @exception_handled
    def on_key_press(self, event: KeyEvent ):
        if event.inaxes == self.ax:
            if event.key == 'backspace': self.delete_selection()

    def delete_selection(self):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        lrec = self.lrecs.pop( self.selected_pid, None )
        if lrec is not None:
            lrec.clear()
            lm().deletePid( self.selected_pid )
            mm().plot_markers_image( clear_highlights=True )
            self.plot(True)

    def remove_region(self, marker: Marker ):
        for pid in marker.pids:
            lrec = self.lrecs.pop( pid, None )
            if lrec is not None:
                lrec.clear()
        self.plot()
        return marker

    def remove_point( self, pid: int ):
        lrec = self.lrecs.pop( pid, None )
        if lrec is not None:
            lrec.clear()
            self.plot()

    @property
    def nlines(self) -> int:
        return len( self.lrecs.keys() )
