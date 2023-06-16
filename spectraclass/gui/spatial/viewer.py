import traceback, time
from typing import List, Union, Tuple, Optional, Dict, Type, Hashable, Callable
import hvplot.xarray
from panel.widgets.player import DiscretePlayer
from spectraclass.learn.cluster.manager import clm
import holoviews as hv
from spectraclass.data.base import dm
from panel.layout import Panel
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.satellite import spm
from holoviews.streams import SingleTap, DoubleTap
import geoviews.feature as gf
import panel as pn
from panel.layout import Panel
import xarray as xa, numpy as np
import os, glob
from enum import Enum
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

class DatasetType(Enum):
    PRODUCT = 1
    METRIC = 2

coastline = gf.coastline.opts(line_color="white", line_width=2.0 ) # , scale='50m')

def arange( data: xa.DataArray, axis=None ) -> Tuple[np.ndarray,np.ndarray]:
    return ( np.nanmin(data.values,axis=axis), np.nanmax(data.values,axis=axis) )

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

def sgui() -> "hvSpectraclassGui":
    return hvSpectraclassGui.instance()

def bounds( data: xa.DataArray ) -> Tuple[ Tuple[float,float], Tuple[float,float] ]:
    xaxis, yaxis = data.coords['x'].values, data.coords['y'].values
    dx, dy = (xaxis[1]-xaxis[0]), (yaxis[1]-yaxis[0])
    xlim = ( xaxis[0]-dx, xaxis[-1]+dx )
    ylim = ( yaxis[0]-dy, yaxis[-1]+dy ) if dy>0 else ( yaxis[-1]+dy, yaxis[0]-dy )
    return xlim,ylim

class VariableBrowser:

    def __init__(self, cname: str, **plotopts ):
        self.cname = cname
        self._block_selection: int = None
        lgm().log( f"Creating VariableBrowser {cname}", print=True)
        self.data: xa.DataArray = sgui().get_data(cname)
        lgm().log(f" --> data shape = {self.data.shape}", print=True)
        self.width = plotopts.get('width', 600)
        self.height = plotopts.get('height', 500)
        self.cmap = plotopts.get('cmap', 'jet')
        self.nIter: int = self.data.shape[0]
        self.player: DiscretePlayer = DiscretePlayer(name='Iteration', options=list(range(self.nIter)), value=self.nIter - 1)
        self.tap_stream = SingleTap( transient=True )
    #    self.double_tap_stream = DoubleTap( transient=True )
        self.selection_dmap = hv.DynamicMap( self.select_points, streams=[self.tap_stream] )
        self.point_graph = hv.DynamicMap( self.update_graph, streams=[self.tap_stream] )
        self.image = hv.DynamicMap(self.get_frame, streams=dict(iteration=self.player.param.value, block_selection=tm().block_selection.param.index))
        self.iter_marker = hv.DynamicMap( self.get_iter_marker, streams=dict( index=self.player.param.value ) )
        self.graph_data = xa.DataArray([])
        self.curves: List[hv.Curve] = []
        self.current_curve_data: Tuple[int,hv.Curve] = None


    # def update_yrange( self, new_range: Tuple[float,float] ):
    #     self.yrange[0] = min( self.yrange[0], new_range[0] )
    #     self.yrange[1] = max( self.yrange[1], new_range[1] )

    @exception_handled
    def select_points(self, x, y):
        lgm().log(f"DYM: select_points")
        ts = time.time()
        if None not in [x, y]:
            lm().on_button_press( x, y )
        t1 = time.time()
        points: List[Tuple[float,float,str]] = lm().getPoints()
        t2 = time.time()
        result = hv.Points(points, vdims='class').opts( marker='+', size=12, line_width=3, angle=45, color='class', cmap=lm().labelmap )
        tf = time.time()
        lgm().log( f"TT: select_points dt={tf-ts} t1={t1-ts} t2={t2-ts}")
        return result

    @exception_handled
    def update_graph(self, x, y) -> hv.Overlay:
        lgm().log(f"DYM: update_graph")
        ts = time.time()
        block: Block = tm().getBlock()
        graph_data: xa.DataArray = self.data.sel(x=x, y=y, method="nearest")
        lgm().log(f"V%% Plotting graph_data[{graph_data.dims}]: shape = {graph_data.shape}, dims={graph_data.dims}, range={arange(graph_data)}")
        is_probe = (lm().current_cid == 0) and (self.cname == "bands")
        line_color = "black" if is_probe else lm().current_color
        popts = dict( width = self.width, height = 200, yaxis = "bare", ylim=(-3,3), alpha=0.6 )
        # if None not in [x, y]:
        #     self.graph_data = self.data.sel(x=x, y=y, method="nearest")
        # elif None not in [x2, y2]:
        #     self.graph_data = self.data.sel(x=x2, y=y2, method="nearest")

        if (self.current_curve_data is not None) and (self.current_curve_data[0] > 0):
            self.curves.append( self.current_curve_data[1].opts(line_width=1) )
        current_curve = hv.Curve(graph_data).opts(line_width=3, line_color = line_color, **popts)
        self.current_curve_data = ( lm().current_cid, current_curve )
        new_curves = [ self.current_curve_data[1] ]
        t1 = time.time()
        if is_probe:
            smean_data: xa.DataArray = dm().modal.getSpectralMean(norm=True)
            lgm().log(f"V%% [{self.cname}] smean_data.shape={smean_data.shape}  graph_data.shape={graph_data.shape}")
            if smean_data.shape[0] == graph_data.shape[0]:
                reproduction: xa.DataArray = block.getReproduction(raster=True)
                verification_data: xa.DataArray = reproduction.sel( x=x, y=y, method="nearest" )
                lgm().log( f"V%% [{self.cname}] verification_data curve{verification_data.dims}, range = {arange(verification_data)}, shape={verification_data.shape}" )
                lgm().log( f"V%% [{self.cname}] smean_data curve{smean_data.dims}, range = {arange(smean_data)}, shape={smean_data.shape}")
                smean_curve        = hv.Curve(    smean_data     ).opts( line_width=1, line_color='red', **popts )
                verification_curve = hv.Curve( verification_data ).opts( line_width=1, line_color='grey', **popts )
                new_curves.extend( [smean_curve,verification_curve] )
        t2 = time.time()
        updated_curves = self.curves + new_curves
        result =  hv.Overlay( updated_curves )
        tf = time.time()
        lgm().log( f"TT: update_graph dt={tf-ts} t1={t1-ts} t2={t2-ts}")
        return result

    def update_block(self, block_selection: int ):
        if self._block_selection != block_selection:
            lgm().log(f" ------------>>>  VariableBrowser[{self.cname}].select_block: {block_selection}  <<<------------ ")
            bcoords = tm().bi2c(block_selection)
            print( f"Select block {bcoords}")
            self.data = sgui().get_data( self.cname, bindex=bcoords )
            self._block_selection = block_selection

    @exception_handled
    def get_frame(self, iteration: int, block_selection: int ):
        lgm().log(f"DYM: get_frame")
        ts = time.time()
        if block_selection >= 0:
            lgm().log( f"Viewer {self.cname}-> get_frame: iteration={iteration} block_selection={block_selection} ")
            self.update_block( block_selection )
        fdata: xa.DataArray = self.data[iteration]
        xlim, ylim = bounds( fdata )
        iopts = dict(width=self.width, height=self.height, cmap=self.cmap, xaxis="bare", yaxis="bare", x="x", y="y", colorbar=False, xlim=xlim, ylim=ylim )
        t2 = time.time()
        result = fdata.hvplot.image( **iopts )
        lgm().log(f"VB: result: {result}")
        tf = time.time()
        lgm().log( f"TT: get_frame dt={tf-ts} t2={t2-ts}")
        return result

    @exception_handled
    def get_iter_marker(self, index: int ):
        lgm().log(f"DYM: get_iter_marker")
        coord: np.ndarray = self.data.coords[ self.data.dims[0] ].values
        vline = hv.VLine( coord[index], label="current iteration")
        return vline.opts( color="grey", alpha=0.5 )

    #  vline = hv.VLine(0.5, label="vline")
    #  curve * vline

    # @exception_handled
    # def plot1(self, **plotopts)-> Panel:
    #     width = plotopts.get('width',600)
    #     widget_type = plotopts.get('widget_type','scrubber')
    #     cmap = plotopts.get('cmap', 'jet')
    #     iopts = dict( width=width, cmap=cmap, xaxis = "bare", yaxis = "bare"  )
    #     return  self.data.hvplot.image( groupby=self.data.dims[0], widget_type=widget_type, widget_location='bottom', **iopts )

    @exception_handled
    def plot(self)-> Panel:
        if self.cname == "bands":
            selector = lm().class_selector
            return pn.Column( selector, self.image*self.selection_dmap, self.player, self.point_graph*self.iter_marker )
        else:
            return pn.Column( self.image*self.selection_dmap, self.player )

class hvSpectraclassGui(SCSingletonConfigurable):

    def __init__(self):
        super(hvSpectraclassGui, self).__init__()
        self.browsers: Dict[str,VariableBrowser] = None
        self.panels: List = None
        self.mapviews: pn.Tabs = None
        self.alert = ufm().gui()

    @exception_handled
    def init( self, **plotopts ):
        collections = [ "bands", 'features', "reproduction" ]
        self.browsers = { cname: VariableBrowser( cname ) for cname in collections }
        self.browsers['bands'].verification = plotopts.pop('verification',None)
        self.panels = [ (cname,browser.plot(**plotopts)) for cname, browser in self.browsers.items() ]
        self.panels.append(('satellite', spm().get_block_basemap()))
        self.panels.append( ('clusters', clm().panel() ) )
        self.mapviews = pn.Tabs( *self.panels, dynamic=True )
        self.tab_watcher = self.mapviews.param.watch(self.on_tab_change, ['active'], onlychanged=True)
        return self

    @exception_handled
    def on_tab_change(self, *events):
        for event in events:
            new_panel = self.mapviews.objects[ event.new ]
            old_panel = self.mapviews.objects[ event.old ]
            if hasattr( new_panel, 'objects' ) and hasattr( old_panel, 'objects' ):
                new_slider: DiscretePlayer = new_panel.objects[2]
                old_slider: DiscretePlayer = old_panel.objects[2]
                if len(new_slider.values) == len(old_slider.values):
                    lgm().log( f" #TC: tab change event, old slider index= {old_slider.value}, new slider index= {new_slider.value}" )
                    new_slider.value.apply.opts( value=old_slider.value )

    def get_data( self, cname: str, **kwargs ) -> xa.DataArray:
        block: Block = tm().getBlock( **kwargs )
        lgm().log( f"sgui:get_data[{cname}] block = {block.index}")
        if cname=="bands":        return block.getBandData(raster=True, norm=True)
        if cname=="features":     return dm().getModelData(block=block, raster=True, norm=True)
        if cname=="reproduction": return block.getReproduction(raster=True)

    def get_control_panel(self) -> Panel:
        from spectraclass.learn.cluster.manager import clm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        data_selection_panel = pn.Tabs(  ("Tile",dm().modal.gui()) ) # , ("Block",dm().modal.gui()) ] )
        manifold_panel = pn.Row( pcm().gui() )
        analytics_gui = pn.Tabs( ("Cluster", clm().gui()) )
        controls = pn.Accordion( ('Data Selection', data_selection_panel ), ('Analytics',analytics_gui), ('Manifold', manifold_panel ), toggle=True, active=[0] )
        return pn.Column( self.alert, controls )

    @exception_handled
    def panel(self, title: str = None, **kwargs ) -> Panel:
        rows = [ self.mapviews ]
        if title is not None: rows.insert( 0, title )
        image_column = pn.Column( *rows )
        result =  pn.Row(  image_column, self.get_control_panel() )
        return result

#
# class VariableBrowser1:
#
#     def __init__(self, data: xa.DataArray, ** plotopts):
#         self.width = plotopts.get('width',600)
#         self.cmap = plotopts.get('cmap', 'jet')
#         self.data: xa.DataArray = data
#         self.nIter = data.shape[0]
# #        self.points = hv.Points([])
#  #       self.clicker = Tap(source=self.points, transient=True, x=np.nan, y=np.nan )
#  #       self.point_selection = hv.DynamicMap(lambda point: hv.Points([point]), streams=[self.clicker])
#         self.player: DiscretePlayer = DiscretePlayer(name='Iteration', options=list(range(self.nIter)), value=self.nIter - 1)
#
#         # @pn.depends(self.clicker.param.x, self.clicker.param.y)
#         # def location(x, y):
#         #     print( f'Click at {x:.2f}, {y:.2f}' )
#  #           return pn.pane.Str(f'Click at {x:.2f}, {y:.2f}', width=200)
#
#
#
#        # @pn.depends(stream.param.x, stream.param.y)
#        # def location(x, y):
#        #     return pn.pane.Str(f'Click at {x:.2f}, {y:.2f}', width=200)
#
# #    def click_callback(self,point):
# #        print( f"Point selected: {point}")
#
#         # dynamic_points = hv.DynamicMap(lambda point: points.select(x=point[0], y=point[1]), streams=[])
#         # dynamic_points.opts(opts.Points(size=10)).redim(x=dim('X Axis'), y=dim('Y Axis'))
#
#     def get_frame(self, iteration: int ):
#         fdata: xa.DataArray = self.data[iteration]
#         iopts = dict(width=self.width, cmap=self.cmap, xaxis="bare", yaxis="bare")
#         return fdata.hvplot( **iopts )
#
#     @exception_handled
#     def plot(self, **plotopts)-> Panel:
#         image = hv.DynamicMap( pn.bind( self.get_frame, iteration=self.player ) )
#         return  pn.Column( image.opts( **plotopts ), self.player )
#
# class RasterCollectionsViewer1:
#
#     def __init__(self, collections: Dict[str,xa.DataArray], **plotopts ):
#         self.browsers = { cname: VariableBrowser( cdata ) for cname, cdata in collections.items() }
#         self.panels = [ (cname,browser.plot(**plotopts)) for cname, browser in self.browsers.items() ]
#
#     def panel(self, title: str = None, **kwargs ) -> Panel:
#         tabs = [ pn.Tabs( *self.panels ) ]
#         if title is not None: tabs.insert( 0, title )
#         background = kwargs.get( 'background', 'WhiteSmoke')
#         return pn.Column( *tabs, background=background )
#
# #if __name__ == '__main__':
