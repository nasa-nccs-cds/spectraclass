import traceback, linecache
from typing import List, Union, Tuple, Optional, Dict, Type, Hashable, Callable
import hvplot.xarray
from panel.widgets.player import DiscretePlayer
import holoviews as hv
from panel.layout import Panel
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.model.labels import LabelsManager, lm
from holoviews.streams import SingleTap, DoubleTap
import geoviews.feature as gf
import panel as pn
from panel.layout import Panel
import xarray as xa, numpy as np
import os, glob
from enum import Enum
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
hv.extension('bokeh')

class DatasetType(Enum):
    PRODUCT = 1
    METRIC = 2

coastline = gf.coastline.opts(line_color="white", line_width=2.0 ) # , scale='50m')

def max_range( current_range: Tuple, series: np.ndarray ) -> Tuple:
    if len(current_range) < 2:  return series[0], series[-1]
    return  min(current_range[0],series[0]), max(current_range[1],series[-1])

def extract_species( data: xa.DataArray, species:str ) -> xa.DataArray:
    result = data.sel(species=species) if 'species' in data.dims else data
    return result.squeeze()

def find_varname( selname: str, varlist: List[str]) -> str:
    for varname in varlist:
        if selname.lower() in varname.lower(): return varname
    raise Exception( f"Unknown variable: '*{selname}', varlist = {varlist}")

class VariableBrowser:

    def __init__(self, data: xa.DataArray, **plotopts ):
        self.data = data
        self.width = plotopts.get('width',600)
        self.cmap = plotopts.get('cmap', 'jet')
        self.nIter = data.shape[0]
        self.player: DiscretePlayer = DiscretePlayer(name='Iteration', options=list(range(self.nIter)), value=self.nIter - 1)
        self.tap_stream = SingleTap( transient=True )
        self.double_tap_stream = DoubleTap( rename={'x': 'x2', 'y': 'y2'}, transient=True)
        self.selection_dmap = hv.DynamicMap(self.select_points, streams=[self.tap_stream, self.double_tap_stream])
        self.point_graph = hv.DynamicMap( self.update_graph, streams=[self.tap_stream, self.double_tap_stream])
        self.image = hv.DynamicMap( pn.bind(self.get_frame, iteration=self.player) )
        self.graph_data = xa.DataArray([])

    @exception_handled
    def select_points(self, x, y, x2, y2):
        if None not in [x, y]:
            lm().on_button_press( x, y )
        elif None not in [x2, y2]:
            lm().on_button_press( x, y )
        points: List[Tuple[float,float,str]] = lm().getPoints()
        return hv.Points(points, vdims='class').opts( marker='+', size=10, color='class', cmap=lm().labelmap )

    @exception_handled
    def update_graph(self, x, y, x2, y2):
        graph_data = self.data.sel(x=x, y=y, method="nearest")
        # if None not in [x, y]:
        #     self.graph_data = self.data.sel(x=x, y=y, method="nearest")
        # elif None not in [x2, y2]:
        #     self.graph_data = self.data.sel(x=x2, y=y2, method="nearest")
        return hv.Curve(graph_data).opts(width=self.width, height=200, yaxis="bare")

      #  return hv.NdOverlay(curves)

    @exception_handled
    def get_frame(self, iteration: int ):
        fdata: xa.DataArray = self.data[iteration]
        iopts = dict(width=self.width, cmap=self.cmap, xaxis="bare", yaxis="bare", colorbar=False)
        return fdata.hvplot.image( **iopts )

    # @exception_handled
    # def plot1(self, **plotopts)-> Panel:
    #     width = plotopts.get('width',600)
    #     widget_type = plotopts.get('widget_type','scrubber')
    #     cmap = plotopts.get('cmap', 'jet')
    #     iopts = dict( width=width, cmap=cmap, xaxis = "bare", yaxis = "bare"  )
    #     return  self.data.hvplot.image( groupby=self.data.dims[0], widget_type=widget_type, widget_location='bottom', **iopts )

    @exception_handled
    def plot(self)-> Panel:
        selector = lm().class_selector
        return pn.Column( selector, self.image*self.selection_dmap, self.player, self.point_graph )

class RasterCollectionsViewer:

    def __init__(self, collections: Dict[str,xa.DataArray], **plotopts ):
        self.browsers = { cname: VariableBrowser( cdata ) for cname, cdata in collections.items() }
        self.panels = [ (cname,browser.plot(**plotopts)) for cname, browser in self.browsers.items() ]
        self.mapviews = pn.Tabs( *self.panels, dynamic=True )
 #       ps, p = self.mapviews.params, self.mapviews.param
 #       print(".")

    def panel(self, title: str = None, **kwargs ) -> Panel:
        rows = [ self.mapviews ]
        if title is not None: rows.insert( 0, title )
        background = kwargs.get( 'background', 'WhiteSmoke')
        return pn.Column( *rows, background=background )

class VariableBrowser1:

    def __init__(self, data: xa.DataArray, ** plotopts):
        self.width = plotopts.get('width',600)
        self.cmap = plotopts.get('cmap', 'jet')
        self.data: xa.DataArray = data
        self.nIter = data.shape[0]
#        self.points = hv.Points([])
 #       self.clicker = Tap(source=self.points, transient=True, x=np.nan, y=np.nan )
 #       self.point_selection = hv.DynamicMap(lambda point: hv.Points([point]), streams=[self.clicker])
        self.player: DiscretePlayer = DiscretePlayer(name='Iteration', options=list(range(self.nIter)), value=self.nIter - 1)

        # @pn.depends(self.clicker.param.x, self.clicker.param.y)
        # def location(x, y):
        #     print( f'Click at {x:.2f}, {y:.2f}' )
 #           return pn.pane.Str(f'Click at {x:.2f}, {y:.2f}', width=200)



       # @pn.depends(stream.param.x, stream.param.y)
       # def location(x, y):
       #     return pn.pane.Str(f'Click at {x:.2f}, {y:.2f}', width=200)

#    def click_callback(self,point):
#        print( f"Point selected: {point}")

        # dynamic_points = hv.DynamicMap(lambda point: points.select(x=point[0], y=point[1]), streams=[])
        # dynamic_points.opts(opts.Points(size=10)).redim(x=dim('X Axis'), y=dim('Y Axis'))

    def get_frame(self, iteration: int ):
        fdata: xa.DataArray = self.data[iteration]
        iopts = dict(width=self.width, cmap=self.cmap, xaxis="bare", yaxis="bare")
        return fdata.hvplot( **iopts )

    @exception_handled
    def plot(self, **plotopts)-> Panel:
        image = hv.DynamicMap( pn.bind( self.get_frame, iteration=self.player ) )
        return  pn.Column( image.opts( **plotopts ), self.player )

class RasterCollectionsViewer1:

    def __init__(self, collections: Dict[str,xa.DataArray], **plotopts ):
        self.browsers = { cname: VariableBrowser( cdata ) for cname, cdata in collections.items() }
        self.panels = [ (cname,browser.plot(**plotopts)) for cname, browser in self.browsers.items() ]

    def panel(self, title: str = None, **kwargs ) -> Panel:
        tabs = [ pn.Tabs( *self.panels ) ]
        if title is not None: tabs.insert( 0, title )
        background = kwargs.get( 'background', 'WhiteSmoke')
        return pn.Column( *tabs, background=background )
