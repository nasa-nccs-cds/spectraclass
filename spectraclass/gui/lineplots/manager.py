import ipywidgets as ip
from abc import ABC, abstractmethod
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

def sel( array: xa.DataArray, pids: Union[int,List[int],np.ndarray,xa.DataArray] ) -> np.ndarray:
    if isinstance( pids, np.ndarray ): pids = pids.tolist()
    elif isinstance( pids, xa.DataArray ): pids = pids.values.tolist()
    elif not isinstance( pids, (list, tuple, set) ): pids = [ pids ]
    lgm().log( f"LinePlot.sel---> array[{array.dims}] shape: {array.shape}, range: {[array.values.min(), array.values.max()]}, pids[:10]={pids[:10]}")
    return array.sel( dict(samples=pids) ).values

class LinePlot(ABC):

    _x: xa.DataArray = None
    _mx: xa.DataArray = None
    _ploty: xa.DataArray = None
    _rploty: xa.DataArray = None
    _mploty: xa.DataArray = None
    _mdata: List[np.ndarray] = None
    _use_model = False

    def __init__( self, index: int, **kwargs ):
        self.index: int = index
        self.standalone: bool = kwargs.pop('standalone', False)
        self.init_data( **kwargs )

    def use_model_data( self, use: bool ):
        self._use_model = use
        self.clear()

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def addMarker( self, m: Marker ):
        pass

    @abstractmethod
    def validate_lines(self, m: Marker):
        pass

    @abstractmethod
    def remove_region(self, m: Marker):
        pass

    @abstractmethod
    def remove_points(self, pids: List[int] ):
        pass

    @classmethod
    def refresh(cls):
        cls._x = None
        cls.init_data()

    @abstractmethod
    def gui(self):
        pass

    @classmethod
    def init_data(cls, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if cls._x is None:
            project_data: Dict[str,Union[xa.DataArray,List,Dict]]  = dm().loadCurrentProject("graph")
            table_cols = DataManager.instance().table_cols
            cls._mdata: List[np.ndarray] = [ cls.get_col_values(project_data[mdv]) for mdv in table_cols ]

            block = tm().getBlock()
            point_data, pcoords = block.getPointData()
            cls._ploty: xa.DataArray = point_data
            cls._x: xa.DataArray = cls._ploty.coords['band']
            cls._rploty: xa.DataArray = block.reproduction
            cls._mploty: xa.DataArray = block.model_data
            cls._mx: xa.DataArray = cls._mploty.coords['band']

    @classmethod
    def get_col_values(cls, data: Union[xa.DataArray,List] ):
        return np.array( data ) if isinstance(data, list) else data.values


    def lx( self, pids: Union[int,List[int]] ) -> np.ndarray:
        xv: xa.DataArray = self._mx if self._use_model else self._x
        if xv.ndim == 1:   return  xv.values
        else:              return  sel( xv, pids ).squeeze()

    def ly( self, pids: Union[int,List[int]] ) -> Optional[np.ndarray]:
        try:
            ydata: xa.DataArray = self._mploty if self._use_model else self._ploty
            return sel( ydata, pids ).squeeze().transpose()
        except KeyError:
            return None

    def lry(self, pid ) -> np.ndarray:
        return sel( self._rploty, pid ).squeeze()

    @property
    def x(self) -> np.ndarray:
        xv = self._mx if self._use_model else self._x
        if xv.ndim == 1:   return  xv
        else:              return  sel( xv, self.pids )

    @property
    def y( self ) -> np.ndarray:
        ydata = self._mploty if self._use_model else self._ploty
        return sel( ydata, self.pids ).transpose()

    @property
    def ry( self ) ->  np.ndarray:
        return sel( self._rploty, self.tpids ).transpose()

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

    @property
    @abstractmethod
    def pids(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def tpids(self) -> List[int]:
        pass


def gpm() -> "GraphPlotManager":
    return GraphPlotManager.instance()

class GraphPlotManager(SCSingletonConfigurable):
    _plottype = "mpl"   #  ds

    def __init__( self ):
        super(GraphPlotManager, self).__init__()
        self._wGui: ipw.Tab() = None
        self._graphs: List[LinePlot] = []
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
        LinePlot.refresh()
        lgm().log(f" GraphPlotManager refresh ")

    def current_graph(self) -> Optional[LinePlot]:
        if self._wGui is None: return None
        return self._graphs[ self._wGui.selected_index ]

    @log_timing
    def plot_graph( self, marker: Marker ):
        current_graph: LinePlot = self.current_graph()
        if current_graph is not None:
            current_graph.addMarker( marker )

    def remove_marker(self, marker: Marker):
        for graph in self._graphs: graph.remove_region( marker )

    def remove_points( self, pids: List[int] ):
        for graph in self._graphs: graph.remove_points( pids )

    def validate_plots( self, m: Marker ):
        for graph in self._graphs: graph.validate_lines( m )

    @classmethod
    def get_graph( cls, index: int, **kwargs ):
        from .mpl import mplGraphPlot
#        from .ds import dsGraphPlot
        if cls._plottype == "mpl": return mplGraphPlot( index, **kwargs )
#        if cls._plottype == "ds":  return dsGraphPlot(  index, **kwargs )

    def _createGui( self, **kwargs ) -> ipw.Tab():
        wTab = ipw.Tab( layout = ip.Layout( width='auto', flex='0 0 500px' ) )
        for iG in range(self._ngraphs):
            self._graphs.append( self.get_graph( iG, **kwargs ) )
            wTab.set_title(iG, str(iG))
        wTab.children = [ g.gui() for g in self._graphs ]
        return wTab

