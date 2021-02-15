from bokeh.plotting import figure
from bokeh.io import output_notebook
import jupyter_bokeh as jbk
from bokeh.transform import linear_cmap
import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable
import xarray as xa
import numpy as np
from spectraclass.data.base import DataManager
from bokeh.models import ColumnDataSource
from spectraclass.util.logs import LogManager, lgm
import ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.model.base import SCSingletonConfigurable

def rescale( x: np.ndarray ):
    xs= x.squeeze()
    return xs / xs.mean()

class JbkPlot:
    _x: np.ndarray = None
    _ploty: np.ndarray = None
    _rploty: np.ndarray = None
    _mdata: List[np.ndarray] = None

    def __init__( self, **kwargs ):
        self.init_data(**kwargs)
        self._selected_pids: List[int] = []
        self._source = None
        self._r = None
        self.init_figure()

    def init_figure(self):
        self.fig = figure(title=self.title, height=250, width=750, background_fill_color='#efefef')
        self._model = jbk.BokehModel( self.fig, layout = ip.Layout( width= 'auto', height= 'auto' ) )
        lgm().log( f"BokehModel: {self._model.keys}" )

    def init_graph(self):
        if self._r is None:
            self._source = ColumnDataSource(data=dict(
                xs=self.x,  # x coords for each line (list of lists)
                ys=self.y,  # y coords for each line (list of lists)
                cmap=[1]  # data to use for colormapping
            ))
            self._r = self.fig.multi_line( 'xs', 'ys', source=self._source, line_color=linear_cmap('cmap', "Turbo256", 0, 255), line_width=1.5, alpha=0.8 )
            lgm().log(f"Creating Graph; x0 shape = {self.x[0].shape},  y0 shape = {self.y[0].shape}")


    def gui(self):
        self.plot()
        return self._model

    @classmethod
    def refresh(cls):
        cls._x = None
        cls.init_data()

    @classmethod
    def init_data(cls, **kwargs ):
        if cls._x is None:
            project_data: xa.Dataset = DataManager.instance().loadCurrentProject("graph")
            cls._x: np.ndarray = project_data["plot-x"].values
            cls._ploty: np.ndarray = project_data["plot-y"].values
            cls._rploty: np.ndarray = project_data["reproduction"].values
            table_cols = DataManager.instance().table_cols
            lgm().log( f" JbkPlot init, using cols {table_cols} from {list(project_data.variables.keys())}, ploty shape = {cls._ploty.shape}, rploty shape = {cls._rploty.shape}" )
            cls._mdata: List[np.ndarray] = [ project_data[mdv].values for mdv in table_cols ]

    def select_items(self, idxs: List[int] ):
        self._selected_pids = idxs

    def plot(self):
        from spectraclass.model.labels import LabelsManager, lm
        self.fig.title.text = self.title
        marked_pids = lm().getPids()
        if lm().current_cid == 0:
            if len(self._selected_pids) > 0:
                self.update_graph( self.x2, self.y2, [0, 100] )
        else:
            if len(marked_pids) > 0:
                self._selected_pids = marked_pids
                self.update_graph( self.x, self.y, np.random.randint( 0, 255, self.nlines ) )

    def update_graph(self, x, y, cmap ):
        self.init_graph()
        self._source.data.update(ys=y, xs=x, cmap=cmap)
        yr = self.yrange
        self.fig.y_range.update( start=yr[0], end=yr[1] )

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
        lgm().log( f" GRAPH:y2-> idx={idx}, val[10] = {rp[:10]} ")
        return [ rescale( self._ploty[idx] ), rp ]

    @property
    def yrange(self):
        ydata: np.ndarray = self._ploty[ self._selected_pids ]
        ymean: np.ndarray = ydata.mean( axis=1 )
 #       lgm().log(f" yrange-> ydata shape={ydata.shape}, ymean shape = {ymean.shape} ")
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
        output_notebook()
        self._wGui: ipw.Tab() = None
        self._graphs: List[JbkPlot] = []
        self._ngraphs = 8

    def gui(self, **kwargs ) -> ipw.Tab():
        if self._wGui is None:
            self._wGui = self._createGui( **kwargs )
        return self._wGui

    def refresh(self):
        JbkPlot.refresh()
        lgm().log(f" GraphPlotManager refresh ")

    def current_graph(self) -> JbkPlot:
        return self._graphs[ self._wGui.selected_index ]

    def plot_graph( self, pids: List[int] = None ):
        from spectraclass.model.labels import LabelsManager, lm
        if self._wGui is not None:
            if pids is None: pids = lm().getPids()
            lgm().log(f" plot spectral graph[{self._wGui.selected_index}]: pids = {pids} ")
            current_graph: JbkPlot = self.current_graph()
            current_graph.select_items( pids )
            current_graph.plot()

    def _createGui( self, **kwargs ) -> ipw.Tab():
        wTab = ipw.Tab( layout = ip.Layout( width='auto', flex='0 0 300px' ) )
        for iG in range(self._ngraphs):
            self._graphs.append(JbkPlot(**kwargs))
            wTab.set_title(iG, str(iG))
        wTab.children = [ g.gui() for g in self._graphs ]
        return wTab

    def on_selection(self, selection_event: Dict ):
        selection = selection_event['pids']
        if len( selection ) > 0:
            lgm().log(f" RAPH.on_selection: nitems = {len(selection)}, pid={selection[0]}")
            self.plot_graph( selection )