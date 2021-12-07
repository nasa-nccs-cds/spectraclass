import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
import xarray as xa
import numpy as np
from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.util.logs import LogManager, lgm, exception_handled
import matplotlib.pyplot as plt
import ipywidgets as ipw
from spectraclass.model.base import SCSingletonConfigurable

def rescale( x: np.ndarray ):
    xs= x.squeeze()
    if xs.mean() == 0.0: return xs
    return xs / xs.mean()

class mplGraphPlot:
    _x: np.ndarray = None
    _ploty: np.ndarray = None
    _rploty: np.ndarray = None
    _mdata: List[np.ndarray] = None

    def __init__( self, index: int, **kwargs ):
        self.index = index
        self.init_data(**kwargs)
        self._selected_pids: List[int] = []
        self.ax : plt.Axes = None
        self.fig : plt.Figure = None
        self.lines: List[plt.Line2D] = []
        self._markers: List[Marker] = []
        self.init_figure()


    def init_figure(self):
        if self.fig is None:
            plt.ioff()
            self.fig: plt.Figure = plt.figure( self.index, figsize = (6, 4) )
            if len(self.fig.axes) == 0: self.fig.add_subplot(111)
            self.ax = self.fig.axes[0]
            self.ax.grid(True)
            self.ax.set_autoscaley_on(True)
            self.ax.set_title(f'Point Spectra {self.index}', fontsize=12)
            plt.ion()

    def gui(self):
        return self.fig.canvas

    @classmethod
    def refresh(cls):
        cls._x = None
        cls.init_data()

    def clear(self):
        self.ax.clear()
        self.lines = []
        self.ax.grid(True)

    @classmethod
    def init_data(cls, **kwargs ):
        if cls._x is None:
            project_data: xa.Dataset = DataManager.instance().loadCurrentProject("graph")
            cls._x: np.ndarray = project_data["plot-x"].values
            cls._ploty: np.ndarray = project_data["plot-y"].values
            cls._rploty: np.ndarray = project_data["reproduction"].values
            table_cols = DataManager.instance().table_cols
            lgm().log( f" mplGraphPlot init, using cols {table_cols} from {list(project_data.variables.keys())}, ploty shape = {cls._ploty.shape}, rploty shape = {cls._rploty.shape}" )
            cls._mdata: List[np.ndarray] = [ project_data[mdv].values for mdv in table_cols ]

    def clear_transients( self, m: Marker ):
        has_transient = (len(self._markers) == 1) and (self._markers[0].cid == 0)
        if has_transient or (m.cid == 0):
            self._markers = []
            self.clear()

    def addMarker( self, m: Marker ):
        self.clear_transients( m )
        self._markers.append( m )
        self._selected_pids = self.get_pids()

    def get_colors(self):
        return sum( [m.colors for m in self._markers], [] )

    def get_pids( self ):
        return sum( [m.pids.tolist() for m in self._markers], [] )

    def plot( self, marker: Marker ):
        self.ax.title.text = self.title
        self.addMarker( marker )
        lgm().log(f"Plotting lines, nselected={len(self._selected_pids)}")
        if marker.cid == 0:
            self.update_graph( self.x2, self.y2 )
        else:
            self.ax.set_prop_cycle( color=self.get_colors() )
            self.update_graph( self.x, self.y )

    def update_graph(self, xs: List[ np.ndarray ], ys: List[ np.ndarray ], **kwargs ):
        for x, y in zip(xs,ys):
            lgm().log( f"Plotting line, xs = {x.shape}, ys = {y.shape}, xrange = {[x.min(),x.max()]}, yrange = {[y.min(),y.max()]}, args = {kwargs}")
            line, = self.ax.plot( x, y, **kwargs )
            self.lines.append(line)
        self.fig.canvas.draw()

    @property
    def nlines(self) -> int:
        return len( self._selected_pids )

    @property
    def x(self) -> List[ np.ndarray ]:
        if self._x.ndim == 1:   return [ self._x ] * self.nlines
        else:                   return [ self._x[ pid ] for pid in self._selected_pids ]

    @property
    def y( self ) -> List[ np.ndarray ]:
        return [ rescale( self._ploty[idx] ) for idx in self._selected_pids ]

    @property
    def ry( self ) -> List[ np.ndarray ]:
        return [ rescale( self._rploty[idx] ) for idx in self._selected_pids ]

    @property
    def x2( self ) -> List[ np.ndarray ]:
        return [ self._x ] * 2 if (self._x.ndim == 1) else [ self._x[self._selected_pids[0]] ] * 2

    @property
    def y2( self ) -> List[ np.ndarray ]:
        idx = self._selected_pids[0]
        rp = rescale( self._rploty[idx] )
        lgm().log( f" GRAPH:y2-> idx={idx}, val[10] = {rp[:10]} ({self._rploty[idx][:10]})")
        return [ rescale( self._ploty[idx] ), rp ]

    @property
    def yrange(self):
        ydata: np.ndarray = self._ploty[ self._selected_pids ]
        ymean: np.ndarray = ydata.mean( axis=1 )
        ys = ydata / ymean.reshape( ymean.shape[0], 1 )
        return ( ys.min(), ys.max() )

    @property
    def title(self ) -> str:
        if len(self._selected_pids) == 1:
            t = ' '.join([str(mdarray[self._selected_pids[0]]) for mdarray in self._mdata])
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
            current_graph.plot( marker )

    def _createGui( self, **kwargs ) -> ipw.Tab():
        wTab = ipw.Tab( layout = ip.Layout( width='auto', flex='0 0 500px' ) )
        for iG in range(self._ngraphs):
            self._graphs.append(mplGraphPlot( iG, **kwargs ))
            wTab.set_title(iG, str(iG))
        wTab.children = [ g.gui() for g in self._graphs ]
        return wTab

