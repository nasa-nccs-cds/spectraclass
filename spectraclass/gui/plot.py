import ipywidgets as ip
from matplotlib.backend_bases import PickEvent, MouseEvent, MouseButton, KeyEvent  # , NavigationToolbar2
from matplotlib.lines import Line2D
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
from collections import OrderedDict
import xarray as xa
import numpy as np
import shapely.vectorized as svect
from spectraclass.data.base import DataManager, dm
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import matplotlib.pyplot as plt
import ipywidgets as ipw
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.widgets.polygon import PolyRec

def rescale( x: np.ndarray ):
    xs= x.squeeze()
    if xs.mean() == 0.0: return xs
    return xs / xs.mean()

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

class mplGraphPlot:
    _x: np.ndarray = None
    _mx: np.ndarray = None
    _ploty: np.ndarray = None
    _rploty: np.ndarray = None
    _mploty: np.ndarray = None
    _mdata: List[np.ndarray] = None
    _use_model = False

    def __init__( self, index: int, **kwargs ):
        self.index: int = index
        self.standalone: bool = kwargs.pop('standalone', False)
        self.rlines: List[Line2D] = []
        self.init_data(**kwargs)
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

    def use_model_data( self, use: bool ):
        self._use_model = use
        self.clear()

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

    @classmethod
    def refresh(cls):
        cls._x = None
        cls.init_data()

    def clear(self, reset: bool = True ):
        for lrec in self.lrecs.values(): lrec.clear()
        for rline in self.rlines: rline.remove()
        self.rlines = []
        if reset: self.lrecs = OrderedDict()

    @classmethod
    def init_data(cls, **kwargs ):
        if cls._x is None:
            project_data: Dict[str,Union[xa.DataArray,List,Dict]]  = dm().loadCurrentProject("graph")
            cls._x: np.ndarray = project_data["plot-x"].values
            cls._mx: np.ndarray = project_data["plot-mx"].values
            cls._ploty: np.ndarray = project_data["plot-y"].values
            cls._rploty: np.ndarray = project_data["reproduction"].values
            cls._mploty: np.ndarray = project_data["reduction"].values
            table_cols = DataManager.instance().table_cols
            cls._mdata: List[np.ndarray] = [ cls.get_col_values(project_data[mdv]) for mdv in table_cols ]

    @classmethod
    def get_col_values(cls, data: Union[xa.DataArray,List] ):
        return np.array( data ) if isinstance(data, list) else data.values

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
    def plot_lines(self, pids: List[int], cid: int ):
        from spectraclass.model.labels import LabelsManager, lm
        color = lm().graph_colors[cid]
        lrecs = [ LineRec(None, pid, cid) for pid in pids ]
        for lrec in lrecs: self.lrecs[lrec.pid] = lrec
        lines = self.ax.plot( self.lx(pids), self.ly(pids), picker=True, pickradius=2, color=color, alpha=1.0, linewidth=1.0 )
        for (lrec,line) in zip(lrecs,lines): lrec.line = line
        self.ax.figure.canvas.draw_idle()

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
        lgm().log( f"Plotting lines, xs = {self.x.shape}, ys = {self.y.shape}, xrange = {[self.x.min(),self.x.max()]}, yrange = {[self.y.min(),self.y.max()]}, args = {kwargs}")
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

    def lx(self, pids: Union[int,List[int]] ) -> np.ndarray:
        xv = self._mx if self._use_model else self._x
        if xv.ndim == 1:   return  xv
        else:              return  xv[pids].squeeze()

    def ly(self, pids: Union[int,List[int]] ) -> np.ndarray:
        ydata = self._mploty[pids] if self._use_model else self._ploty[pids]
        return self.normalize( ydata ).squeeze().transpose()

    def lry(self, pid ) -> np.ndarray:
        ydata = self._rploty[ pid ]
        return self.normalize( ydata ).squeeze()

    @property
    def x(self) -> np.ndarray:
        xv = self._mx if self._use_model else self._x
        if xv.ndim == 1:   return  xv
        else:              return  xv[ self.pids ]

    @property
    def y( self ) -> np.ndarray :
        ydata = self._mploty[self.pids] if self._use_model else self._ploty[self.pids]
        return self.normalize( ydata ).transpose()

    @property
    def ry( self ) ->  np.ndarray:
        ydata = self._rploty[ self.tpids ]
        return self.normalize( ydata ).transpose()

    def normalize(self, data: np.ndarray ) -> np.ndarray:
        axis = 1 if (data.ndim > 1) else 0
        mean, std = data.mean( axis=axis, keepdims=True ), data.std( axis=axis, keepdims=True )
        return ( data - mean) / std

    @property
    def title(self ) -> str:
        if len(self.pids) == 1:
            t = ' '.join([str(mdarray[self.pids[0]]) for mdarray in self._mdata])
        else:
            t = "multiplot"
        return t

def gpm() -> "GraphPlotManager":
    return GraphPlotManager.instance()

class GraphPlotManager(SCSingletonConfigurable):

    def __init__( self ):
        super(GraphPlotManager, self).__init__()
        self._wGui: ipw.Tab() = None
        self._graphs: List[mplGraphPlot] = []
        self._ngraphs = 8

    def gui(self, **kwargs ) -> ipw.Tab():
        if self._wGui is None:
            self._wGui = self._createGui( **kwargs )
        return self._wGui

    def use_model_data(self, use: bool ):
        for g in self._graphs: g.use_model_data( use )

    def clear(self):
        for g in self._graphs:
            g.clear()

    def refresh(self):
        mplGraphPlot.refresh()
        lgm().log(f" GraphPlotManager refresh ")

    def current_graph(self) -> Optional[mplGraphPlot]:
        if self._wGui is None: return None
        return self._graphs[ self._wGui.selected_index ]

    @exception_handled
    def plot_graph( self, marker: Marker ):
        current_graph: mplGraphPlot = self.current_graph()
        if current_graph is not None:
            current_graph.addMarker( marker )

    def remove_marker(self, marker: Marker):
        for graph in self._graphs: graph.remove_region( marker )

    def remove_point( self, pid: int ):
        for graph in self._graphs: graph.remove_point( pid )

    def _createGui( self, **kwargs ) -> ipw.Tab():
        wTab = ipw.Tab( layout = ip.Layout( width='auto', flex='0 0 500px' ) )
        for iG in range(self._ngraphs):
            self._graphs.append(mplGraphPlot( iG, **kwargs ))
            wTab.set_title(iG, str(iG))
        wTab.children = [ g.gui() for g in self._graphs ]
        return wTab

