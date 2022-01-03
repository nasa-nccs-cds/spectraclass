import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
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

class mplGraphPlot:
    _x: np.ndarray = None
    _mx: np.ndarray = None
    _ploty: np.ndarray = None
    _rploty: np.ndarray = None
    _mploty: np.ndarray = None
    _mdata: List[np.ndarray] = None
    _use_model = False

    def __init__( self, index: int, **kwargs ):
        self.index = index
        self.standalone = kwargs.pop('standalone', False)
        self.init_data(**kwargs)
        self._selected_pids: List[int] = []
        self.ax : plt.Axes = None
        self.fig : plt.Figure = None
        self.lines: plt.Line2D = None
        self._markers: List[Marker] = []
        self._regions: Dict[int,Marker] = {}
        self._points = Dict[int, Marker]
        self.init_figure( **kwargs )

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
            if not self.standalone: plt.ion()

    def gui(self):
        return self.fig.canvas

    @classmethod
    def refresh(cls):
        cls._x = None
        cls.init_data()

    def clear(self):
        self.ax.clear()
        self.lines = None
        self.ax.grid(True)

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
        from shapely.geometry import Polygon
        raster:  xa.DataArray = tm().getBlock().data[0].squeeze()
        X, Y = raster.x.values, raster.y.values
        polygon = Polygon(prec.poly.get_xy())
        MX, MY = np.meshgrid(X, Y)
        PID = np.array(range(raster.size))
        mask = svect.contains( polygon, MX, MY )
        pids = PID[mask.flatten()].tolist()
        marker =  Marker( "label", pids, cid )
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

    def clear_transients( self, m: Marker ):
        has_transient = (len(self._markers) == 1) and (self._markers[0].cid == 0)
        if has_transient or (m.cid == 0):
            self._markers = []

    @log_timing
    def addMarker( self, m: Marker ):
        self.clear_transients( m )
        self._markers.append( m )
        self._selected_pids = self.get_pids()
        self.plot()

    def get_colors(self):
        return sum( [m.colors for m in self._markers], [] )

    def get_pids( self ):
        return sum( [m.pids.tolist() for m in self._markers], [] )

    def plot( self ):
        self.clear()
        self.ax.title.text = self.title
        nsel = len(self._selected_pids)
        if nsel == 1:
            self.update_graph( self.x, self.y2 )
        elif nsel > 1:
            self.ax.set_prop_cycle( color=self.get_colors() )
            self.update_graph( self.x, self.y )

    def update_graph(self, x:  np.ndarray, y: np.ndarray, **kwargs ):
        lgm().log( f"Plotting lines, xs = {x.shape}, ys = {y.shape}, xrange = {[x.min(),x.max()]}, yrange = {[y.min(),y.max()]}, args = {kwargs}")
        self.lines, = self.ax.plot( x, y, **kwargs )
        self.fig.canvas.draw()

    @property
    def nlines(self) -> int:
        return len( self._selected_pids )

    @property
    def x(self) -> np.ndarray:
        xv = self._mx if self._use_model else self._x
        if xv.ndim == 1:   return  xv
        else:              return  xv[ self._selected_pids ]

    @property
    def ydata( self )  -> np.ndarray:
        return self._mploty[self._selected_pids] if self._use_model else self._ploty[self._selected_pids]

    @property
    def y( self ) -> np.ndarray :
        return self.ydata

    @property
    def ry( self ) ->  np.ndarray:
        return  self._rploty[self._selected_pids]

    @property
    def y2( self ) -> np.ndarray:
        idx = self._selected_pids[0]
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

