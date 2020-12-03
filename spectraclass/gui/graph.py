from bokeh.plotting import figure
from bokeh.io import output_notebook
import jupyter_bokeh as jbk
from bokeh.transform import linear_cmap
import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable
import xarray as xa
import numpy as np
from spectraclass.data.manager import DataManager
from bokeh.models import ColumnDataSource
import ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.model.base import AstroConfigurable

class JbkGraph:
    _x: np.ndarray = None
    _ploty: np.ndarray = None
    _rploty: np.ndarray = None
    _mdata: List[np.ndarray] = None

    def __init__( self, **kwargs ):
        self.init_data(**kwargs)
        self._selected_pids: List[int] = [0]
        self._source = ColumnDataSource(data=dict(
            xs=self.x,  # x coords for each line (list of lists)
            ys=self.y,  # y coords for each line (list of lists)
            cmap=[1]  # data to use for colormapping
        ))
        self.fig = figure( title=self.title, height=300, width=1000, background_fill_color='#efefef' )
        self._r = self.fig.multi_line( 'xs', 'ys', source=self._source, line_color=linear_cmap('cmap', "Turbo256", 0, 255), line_width=1.5, alpha=0.8 )
    #    print(f"Creating BokehModel; x0 shape = {self.x[0].shape},  y0 shape = {self.y[0].shape}")
        self._model = jbk.BokehModel( self.fig, layout = ip.Layout( width= 'auto', height= 'auto' ) )
    #    print( f"BokehModel: {self._model.keys}" )

    def gui(self):
        self.plot()
        return self._model

    @classmethod
    def refresh(cls):
        cls._x = None
        cls._ploty = None
        cls._rploty  = None
        cls._mdata = None

    @classmethod
    def init_data(cls, **kwargs ):
        if cls._x is None:
            project_data: xa.Dataset = DataManager.instance().loadCurrentProject("graph")
            cls._x: np.ndarray = project_data["plot-x"].values
            cls._ploty: np.ndarray = project_data["plot-y"].values
            cls._rploty: np.ndarray = project_data["reproduction"].values
            table_cols = DataManager.instance().table_cols
    #        print( f"           &&&&   JbkGraph init, using cols {table_cols} from {list(project_data.variables.keys())}, ploty shape = {cls._ploty.shape}" )
            cls._mdata: List[np.ndarray] = [ project_data[mdv].values for mdv in table_cols ]

    def select_items(self, idxs: List[int] ):
        self._selected_pids = idxs

    def plot(self):
        y, yr = self.y, self.yrange
        self.fig.title.text = self.title
        if self.nlines == 1:
            self._source.data.update( ys=self.y2, xs=self.x2, cmap=[1, 254] )
        else:
            self._source.data.update( ys = y, xs=self.x, cmap = np.random.randint( 0, 255, self.nlines ) )
        self.fig.y_range.update( start=yr[0], end=yr[1] )
    #    print( f"           &&&&   GRAPH:plot-> title={self.title}, nlines={nlines}, y0 shape = {y[0].shape}, x0 shape = {self.x[0].shape}")

    @property
    def nlines(self) -> int:
        return len( self._selected_pids )

    @property
    def x(self) -> List[ np.ndarray ]:
        if self._x.ndim == 1:   return [ self._x ] * self.nlines
        else:                   return [ self._x[ pid ] for pid in self._selected_pids ]

    @property
    def y( self ) -> List[ np.ndarray ]:
        return [ self._ploty[idx].squeeze() for idx in self._selected_pids ]

    @property
    def ry( self ) -> List[ np.ndarray ]:
        return [ self._rploty[idx].squeeze() for idx in self._selected_pids ]

    @property
    def x2( self ) -> List[ np.ndarray ]:
        idx = self._selected_pids[0]
        return [ self._x[ idx ] ] * 2

    @property
    def y2( self ) -> List[ np.ndarray ]:
        idx = self._selected_pids[0]
        print( f"           &&&&   GRAPH:y2-> idx={idx} ")
        return [ self._ploty[idx].squeeze(), self._rploty[idx].squeeze() ]

    @property
    def yrange(self):
        ydata = self._ploty[ self._selected_pids ]
        return ( ydata.min(), ydata.max() )

    @property
    def title(self ) -> str:
        if len(self._selected_pids) == 1:
            t = ' '.join([str(mdarray[self._selected_pids[0]]) for mdarray in self._mdata])
        else:
            t = "multiplot"
        return t

class GraphManager(tlc.SingletonConfigurable, AstroConfigurable):

    def __init__( self ):
        super(GraphManager, self).__init__(  )
        output_notebook()
        self._wGui: ipw.Tab() = None
        self._graphs: List[JbkGraph] = []
        self._ngraphs = 8

    def gui(self, **kwargs ) -> ipw.Tab():
        if self._wGui is None:
            self._wGui = self._createGui( **kwargs )
        return self._wGui

    def refresh(self):
        JbkGraph.refresh()
        print(f"           &&&&   GraphManager refresh ")
        self._wGui = None

    def current_graph(self) -> JbkGraph:
        return self._graphs[ self._wGui.selected_index ]

    def plot_graph( self, pids: List[int] ):
        print(f"           &&&&   plot_graph[{self._wGui.selected_index}]: pids = {pids} ")
        current_graph: JbkGraph = self.current_graph()
        current_graph.select_items( pids )
        current_graph.plot()

    def _createGui( self, **kwargs ) -> ipw.Tab():
        wTab = ipw.Tab( layout = ip.Layout( width='auto', flex='0 0 330px' ) )
        for iG in range(self._ngraphs):
            self._graphs.append( JbkGraph( **kwargs ) )
            wTab.set_title(iG, str(iG))
        wTab.children = [ g.gui() for g in self._graphs ]
        return wTab

    def on_selection(self, selection_event: Dict ):
        selection = selection_event['pids']
        if len( selection ) > 0:
    #        print(f"           &&&&   GRAPH.on_selection: nitems = {len(selection)}, pid={selection[0]}")
            self.plot_graph( selection )