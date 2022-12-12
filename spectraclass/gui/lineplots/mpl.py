from matplotlib.backend_bases import PickEvent, MouseEvent, MouseButton, KeyEvent  # , NavigationToolbar2
from matplotlib.lines import Line2D
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
from collections import OrderedDict
import time, xarray as xa
import numpy as np
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import matplotlib.pyplot as plt
from .manager import LinePlot
import ipywidgets as ipw

def sel( array: xa.DataArray, pids: List[int] ) -> xa.DataArray:
    mask: np.ndarray = np.isin( array.samples.values, np.array(pids) )
    return array[mask,:]

class LineRec:

    def __init__(self, line: Optional[Line2D], pid: int, cid: int, marker: Marker ):
        self.line: Optional[Line2D] = line
        self.pid: int = pid
        self.cid: int = cid
        self._marker = marker

    def set_width(self, lw: float ):
        self.line.set_linewidth(lw)

    def set_alpha(self, a: float ):
        self.line.set_alpha(a)

    @property
    def mpids(self) -> List[int]:
        return [self.pid] if self._marker is None else self._marker.gids.tolist()

    @property
    def marker(self) -> Marker:
        return self._marker

    def validate(self, m: Marker) -> bool:
        return (self._marker != m) or (self.pid in m.gids)

    def clear(self):
        if self.line is not None:
            self.line.remove()
            self.line = None

    @property
    def id(self) -> int:
        return -1 if self.line is None else self.lid(self.line)

    @classmethod
    def lid( cls, line: Line2D ) -> int:
        return id(line)

class mplGraphPlot(LinePlot):

    def __init__( self, index: int, **kwargs ):
        LinePlot.__init__( self, index, **kwargs )
        self.rlines: List[Line2D] = []
        self._max_graph_group_size = 100
        self.ax: plt.Axes = None
        self.fig: plt.Figure = None
        self.selected_pid: int = -1
        self.lrecs: OrderedDict[int, LineRec] = OrderedDict()
        self.marked_lrecs: Dict[ str, List[int] ] = {}
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
            self.fig: plt.Figure = plt.figure( self.index+10, figsize = (6, 4) )
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
        lgm().log(f"mplGraphPlot: Add Marker[{m.size}]: cid={m.cid}, pids[:10]={m.gids[:10]}")
        if m.size > 0:
            self.clearTransients()
            if len(m.gids) == 1:    self.highlight_line(m)
            else:                   self.plot_lines( m )

    @log_timing
    def highlight_line(self, m: Marker):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        cid, pid = m.cid, m.gids[0]
        self.selected_pid = pid
        alpha, lw = 1.0, 3.0
        color = lm().graph_colors[cid]
        x,y = self.lx(pid), self.ly(pid)
        if y is not None:
            for lrec in self.lrecs.values():
                lrec.set_width(1.0)
                lrec.set_alpha(0.5)
            lines = self.ax.plot( x, y, picker=True, pickradius=2, color=color, alpha=alpha, linewidth=lw )
            self.lrecs[pid] = LineRec(lines[0], pid, cid, m)
            self.marked_lrecs[m.oid] = [pid]
            if (not self._use_model) and (self.ry.size > 0):
                self.rlines.extend( self.ax.plot( x, self.lry(pid), color="grey" ) )
            self.ax.figure.canvas.draw_idle()
        else:
            ufm().show(f"Points out of bounds","red")

    def subset(self):
        pass

    @log_timing
    def plot_lines(self, m: Marker ):    # emphasize anomalies with subset
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        cid: int = m.cid
        color = lm().graph_colors[cid]
        skip_index = max(m.gids.size // self._max_graph_group_size, 1)
        pids = m.gids[::skip_index] if len(m.gids) > skip_index else m.gids
        x,y = self.lx(pids), self.ly(pids)
        if y is not None:
            lines = self.ax.plot(x, y, picker=True, pickradius=2, color=color, alpha=0.2, linewidth=1.0)
            self.marked_lrecs[m.oid] = pids.tolist()
            for pid,line in zip(pids,lines):
                self.lrecs[pid] = LineRec( line, pid, cid, m )
            self.ax.figure.canvas.draw_idle()
        else:
            ufm().show(f"Points out of bounds","red")

    def expose_nearby_lines(self, pid: int, mpids: List[int], cid: int, eps = 0.05 ):
        target_line, line_group = sel(self._ploty,[pid]), sel(self._ploty,mpids)
        diff: np.array = np.abs( target_line.values - line_group.values )
        offset: np.array = diff.sum(axis=1)
        idmask = (offset > 0.0)
        minval, maxval = offset[idmask].min(), offset.max()
        lgm().log(f"\n              expose_nearby_lines[{pid}]: cid={cid}, offset.shape={offset.shape}, offset.range={[minval,maxval]},")
        thresh = minval + (maxval-minval)*eps
        mask = (offset < thresh) & idmask
        pids = np.array(mpids)[ mask ].tolist()
        lgm().log(f"                 ---> thresh={thresh}, mask-nz={np.count_nonzero(mask)}, pids={pids}")
        self.plot_lines( pids, cid )


    @exception_handled
    def get_plotspecs(self):
        from spectraclass.model.labels import LabelsManager, lm
        colors, alphas, lws = [], [], []
#        lgm().log( f"create plotspecs for {len(self.lrecs.items())} lines, pids({self.selected_pid})->{self.pids}")
        for (pid, lrec) in self.lrecs.items():
            selection = ( pid == self.selected_pid )
            alphas.append( 1.0 if selection else 0.2 )
            colors.append( lm().graph_colors[ lrec.cid ] )
            lws.append( 2.0 if selection else 1.0 )
        return dict( color=colors, alpha=alphas, lw=lws )

    @property
    def cids( self ):
        return  [ lrec.cid for lrec in self.lrecs.values() ]

    @log_timing
    def plot( self, clear_selection = False ):
        self.ax.title.text = self.title
        if clear_selection: self.selected_pid = -1
        ps = self.get_plotspecs()
        colors, alphas, linewidths = ps['color'], ps['alpha'], ps['lw']
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
        if selected_lrec is not None:
            self.selected_pid = selected_lrec.pid
            mm().highlight_points( [self.selected_pid], [selected_lrec.cid] )
            self.plot()

    @exception_handled
    def mark_selection( self, *args ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.model.labels import LabelsManager, lm
        cid = lm().current_cid
        lgm().log(f"mark_selection: pid={self.selected_pid} cid={cid}", print=True )
        if self.selected_pid >= 0:
            lrec: LineRec = self.lrecs.pop( self.selected_pid )
            lrec.clear()
            mm().mark_point( lrec.pid, cid=cid )
            self.expose_nearby_lines( lrec.pid, lrec.mpids, cid )
            mm().plot_markers_image( clear_highlights=True )
            lgm().log(f"MARK SELECTION: cid={cid}, pid={lrec.pid}")
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

    def remove_points( self, pids: List[int], **kwargs ):
        plot = kwargs.get('plot',False)
        for pid in pids:
            lrec = self.lrecs.pop( pid, None )
            if lrec is not None:
                lrec.clear()
        if plot: self.plot()

    def remove_marker(self, m: Marker, **kwargs ):
        pids = self.marked_lrecs.pop(m.oid,[])
        if len(pids):
            lgm().log( f"Remove marker[{m.cid}]: {len(pids)} pids")
            self.remove_points( pids, **kwargs )

    @property
    def nlines(self) -> int:
        return len( self.lrecs.keys() )
