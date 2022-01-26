import ipywidgets as ip
from matplotlib.backend_bases import PickEvent, MouseEvent, MouseButton, KeyEvent  # , NavigationToolbar2
from matplotlib.lines import Line2D
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
from collections import OrderedDict
import xarray as xa
import numpy as np
import shapely.vectorized as svect
from spectraclass.data.base import DataManager
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

    def remove(self):
        self.line.remove()

    @property
    def id(self) -> int:
        return id(self.line)

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
        self.init_data(**kwargs)
        self.ax: plt.Axes = None
        self.fig: plt.Figure = None
        self.selected_pid: int = -1
        self.lrecs: OrderedDict[int, LineRec] = OrderedDict()
        self._markers: List[Marker] = []
        self._regions: Dict[int,Marker] = {}
        self.init_figure( **kwargs )

    def get_lrec( self, id: int ) -> Optional[LineRec]:
        for lrec in self.lrecs.values():
            if lrec.id == id: return lrec
        return None

    def get_selected_lrec( self ) -> Optional[LineRec]:
        if self.selected_pid == -1: return None
        return self.lrecs[ self.selected_pid ]

    @property
    def pids(self) -> List[int]:
        return list( self.lrecs.keys() )

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

    def clear(self):
        for line in self.lrecs.values(): line.remove()
        self.lrecs = OrderedDict()

    @classmethod
    def init_data(cls, **kwargs ):
        if cls._x is None:
            project_data: xa.Dataset = DataManager.instance().loadCurrentProject("graph")
            cls._x: np.ndarray = project_data["plot-x"].values
            cls._mx: np.ndarray = project_data["plot-mx"].values
            cls._ploty: np.ndarray = project_data["plot-y"].values
            cls._rploty: np.ndarray = project_data["reproduction"].values
            cls._mploty: np.ndarray = project_data["reduction"].values
            table_cols = DataManager.instance().table_cols
            lgm().log( f" mplGraphPlot init, using cols {table_cols} from {list(project_data.variables.keys())}, ploty shape = {cls._ploty.shape}, rploty shape = {cls._rploty.shape}" )
            cls._mdata: List[np.ndarray] = [ project_data[mdv].values for mdv in table_cols ]

    @exception_handled
    @log_timing
    def get_region_marker(self, prec: PolyRec, cid: int ) -> Marker:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.spatial.tile.tile import Block, Tile
        from shapely.geometry import Polygon
        block: Block = tm().getBlock()
        idx2pid: np.ndarray = block.index_array.values.flatten()
        raster:  xa.DataArray = block.data[0].squeeze()
        X, Y = raster.x.values, raster.y.values
        polygon = Polygon(prec.poly.get_xy())
        MX, MY = np.meshgrid(X, Y)
        PID: np.ndarray = np.array(range(raster.size))
        mask: np.ndarray = svect.contains( polygon, MX, MY ).flatten()
        pids = idx2pid[ PID[mask] ]
        marker = Marker( "label", pids[ pids > -1 ].tolist(), cid )
        self._regions[ prec.polyId ] = marker
        return marker

    def plot_region(self, prec: PolyRec, cid: int ) -> Marker:
        marker = self.get_region_marker( prec, cid )
        self.addMarker( marker )
        return marker

    def removeMarker(self, marker: Marker ):
        if marker is not None:
            try:
                self._markers.remove(marker)
                self._selected_pids = self.get_pids()
                self._selected_cids = self.get_cids()
                self.plot()
            except:
                lgm().log( f"Error in removeMarker: #markers={len(self._markers)}, marker = {marker}")

    def remove_region(self, prec: PolyRec ):
        marker = self._regions.pop( prec.polyId, None )
        self.removeMarker( marker )

    def remove_point(self, pid: int):
        for marker in self._markers:
            if marker.empty or (marker.pids[0] == pid):
                self.removeMarker(marker)

    @log_timing
    def addMarker( self, m: Marker ):
        self._markers.append( m )
        for pid in m.pids:
            self.lrecs[pid] = LineRec( None, pid, m.cid )
        lgm().log(f"Add Marker: cid={m.cid}, pids = {m.pids}")
        self.plot()

    def get_plotspecs(self):
        from spectraclass.model.labels import LabelsManager, lm
        colors, alphas, lws = [], [], []
        selected: bool = (self.selected_pid >= 0)
        for (pid, lrec) in self.lrecs.items():
            selection = ( pid == self.selected_pid )
            alphas.append( 0.2 if (selection and not selected) else 1.0 )
            colors.append( lm().graph_colors[ lrec.cid ] )
            lws.append( 2.0 if selected else 1.0 )
        return dict( color=colors, alpha=alphas, lw=lws)

    def get_pids( self ):
        return sum( [m.pids.tolist() for m in self._markers], [] )

    def get_cids( self ):
        return sum( [ [m.cid]*m.size for m in self._markers ], [] )

    def plot( self ):
        self.ax.title.text = self.title
        ps = self.get_plotspecs()
        self.ax.set_prop_cycle( color=ps['color'], alpha=ps['alpha'], linewidth=ps['lw'] )
        self.update_graph()

    def update_graph(self, **kwargs ):
        lgm().log( f"Plotting lines, xs = {self.x.shape}, ys = {self.y.shape}, xrange = {[self.x.min(),self.x.max()]}, yrange = {[self.y.min(),self.y.max()]}, args = {kwargs}")
        lines: List[Line2D] = self.ax.plot( self.x, self.y, picker=True, pickradius=2, **kwargs )
        for (line, lrec) in zip(lines, self.lrecs.values()): lrec.line = line
        self.fig.canvas.draw()

    @exception_handled
    def onpick(self, event: PickEvent ):
        line: Line2D = event.artist
        selected_lrec = self.get_lrec( id(line) )
        self.selected_pid = selected_lrec.pid
        self.plot()

    @exception_handled
    def on_key_press(self, event: KeyEvent ):
        if event.inaxes == self.ax:
            if event.key == 'backspace': self.delete_selection()

    def delete_selection(self):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        if self.selected_pid >= 0:
            lrec = self.lrecs.pop( self.selected_pid )
            lrec.remove()
            lm().deletePid( self.selected_pid )
            mm().plot_markers_image()
            self.plot()

    @property
    def nlines(self) -> int:
        return len( self.lrecs.keys() )

    @property
    def x(self) -> np.ndarray:
        xv = self._mx if self._use_model else self._x
        if xv.ndim == 1:   return  xv
        else:              return  xv[ self.pids ]

    @property
    def ydata( self )  -> np.ndarray:
        return self._mploty[self.pids] if self._use_model else self._ploty[self.pids]

    @property
    def y( self ) -> np.ndarray :
        return self.ydata.transpose()

    @property
    def ry( self ) ->  np.ndarray:
        return  self._rploty[self.pids]

    @property
    def y2( self ) -> np.ndarray:
        idx = self.pids[0]
        if self._use_model: return self._mploty[idx]
        else:               return np.concatenate( [ np.expand_dims(self._ploty[idx],1), np.expand_dims(self._rploty[idx],1) ], axis=1 )

    @property
    def yrange(self):
        ydata: np.ndarray = self.ydata
        ymean: np.ndarray = ydata.mean( axis=1 )
        ys = ydata / ymean.reshape( ymean.shape[0], 1 )
        return ( ys.min(), ys.max() )

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

    @exception_handled
    @log_timing
    def plot_region( self, region: PolyRec, cid: int ) -> Marker:
        current_graph: mplGraphPlot = self.current_graph()
        if current_graph is not None:
            return current_graph.plot_region( region, cid )

    def remove_region(self, region: PolyRec ):
        for graph in self._graphs: graph.remove_region( region )

    def remove_point( self, pid: int ):
        for graph in self._graphs: graph.remove_point( pid )

    def _createGui( self, **kwargs ) -> ipw.Tab():
        wTab = ipw.Tab( layout = ip.Layout( width='auto', flex='0 0 500px' ) )
        for iG in range(self._ngraphs):
            self._graphs.append(mplGraphPlot( iG, **kwargs ))
            wTab.set_title(iG, str(iG))
        wTab.children = [ g.gui() for g in self._graphs ]
        return wTab

