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

class LinePlot:

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
        self.init_data( **kwargs )

    def abstract(self):
        raise NotImplementedError("Calling unimplemented method on abstract baseclass LinePlot")

    def use_model_data( self, use: bool ):
        self._use_model = use
        self.clear()

    def clear(self):
        self.abstract()

    def addMarker( self, m: Marker ):
        self.abstract()

    def remove_region(self, m: Marker):
        self.abstract()

    def remove_point(self, pid: int ):
        self.abstract()

    @classmethod
    def refresh(cls):
        cls._x = None
        cls.init_data()

    def gui(self):
        self.abstract()

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

def gpm() -> "GraphPlotManager":
    return GraphPlotManager.instance()

class GraphPlotManager(SCSingletonConfigurable):
    _plottype = "mpl"

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

    @exception_handled
    def plot_graph( self, marker: Marker ):
        current_graph: LinePlot = self.current_graph()
        if current_graph is not None:
            current_graph.addMarker( marker )

    def remove_marker(self, marker: Marker):
        for graph in self._graphs: graph.remove_region( marker )

    def remove_point( self, pid: int ):
        for graph in self._graphs: graph.remove_point( pid )

    @classmethod
    def get_graph( cls, index: int, **kwargs ):
        from .mpl import mplGraphPlot
        if cls._plottype == "mpl": return mplGraphPlot( index, **kwargs )

    def _createGui( self, **kwargs ) -> ipw.Tab():
        wTab = ipw.Tab( layout = ip.Layout( width='auto', flex='0 0 500px' ) )
        for iG in range(self._ngraphs):
            self._graphs.append( self.get_graph( iG, **kwargs ) )
            wTab.set_title(iG, str(iG))
        wTab.children = [ g.gui() for g in self._graphs ]
        return wTab

