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

class LineRec:

    def __init__(self, line: Optional[Line2D], pid: int, cid: int ):
        self.line: Optional[Line2D] = line
        self.pid: int = pid
        self.cid: int = cid

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
        return self.fig.canvas

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
            else:                   self.plot_lines( m.pids, m.cid )

    @log_timing
    def plot_line(self, pid: int, cid: int ):
        from spectraclass.model.labels import LabelsManager, lm
        selected: bool = (self.selected_pid >= 0)
        selection = (pid == self.selected_pid)
        alpha = (1.0 if selection else 0.2) if selected else 1.0
        color = lm().graph_colors[cid]
        lw = 2.0 if selection else 1.0
        lrec = LineRec(None, pid, cid)
        self.lrecs[pid] = lrec
        lines = self.ax.plot( self.lx(lrec.pid), self.ly(lrec.pid), picker=True, pickradius=2, color=color, alpha=alpha, linewidth=lw )
        lrec.line = lines[0]
        if (not self._use_model) and (self.ry.size > 0):
            self.rlines.extend( self.ax.plot( self.lx(lrec.pid), self.lry(lrec.pid), color="grey" ) )
        self.ax.figure.canvas.draw_idle()

    @log_timing
    def plot_lines(self, mpids: List[int], cid: int ):
        from spectraclass.model.labels import LabelsManager, lm
        color = lm().graph_colors[cid]
        skip_index = max( len(mpids)//self._max_graph_group_size, 1 )
        pids = mpids[::skip_index]
        lrecs = [ LineRec(None, pid, cid) for pid in pids ]
        for lrec in lrecs: self.lrecs[lrec.pid] = lrec
        lines = self.ax.plot( self.lx(pids), self.ly(pids), picker=True, pickradius=2, color=color, alpha=0.2, linewidth=1.0 )
        self.ax.figure.canvas.draw_idle()
        for (lrec, line) in zip(lrecs, lines): lrec.line = line

    @exception_handled
    def get_plotspecs(self):
        from spectraclass.model.labels import LabelsManager, lm
        colors, alphas, lws = [], [], []
        selected: bool = (self.selected_pid >= 0)
     #   lgm().log( f"create plotspecs for {len(self.lrecs.items())} lines, pids({self.selected_pid})->{self.pids}")
        for (pid, lrec) in self.lrecs.items():
            selection = ( pid == self.selected_pid )
            alphas.append( (1.0 if selection else 0.2) if selected else 1.0 )
            colors.append( lm().graph_colors[ lrec.cid ] )
            lws.append( 2.0 if selection else 1.0 )
        return dict( color=colors, alpha=alphas, lw=lws)

    @property
    def cids( self ):
        return  [ lrec.cid for lrec in self.lrecs.values() ]

    @log_timing
    def plot( self, clear_selection = False ):
        self.ax.title.text = self.title
        if clear_selection: self.selected_pid = -1
        ps = self.get_plotspecs()
        try:
            self.ax.set_prop_cycle( color=ps['color'], alpha=ps['alpha'], linewidth=ps['lw'] )
        except Exception as err:
            lgm().log(f"set_prop_cycle: color={ps['color']}, alpha={ps['alpha']}, linewidth={ps['lw']}")
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
