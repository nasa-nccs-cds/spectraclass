import traceback, time
from typing import List, Union, Tuple, Optional, Dict, Type, Hashable, Callable
import hvplot.xarray
import traitlets as tl
from holoviews.streams import Stream, param
from panel.widgets import IntSlider
from spectraclass.learn.cluster.manager import clm
import holoviews as hv
import traitlets.config as tlc
from panel.layout import Panel
from spectraclass.widgets.masks import mm, MaskManager
from spectraclass.widgets.regions import RegionSelector, rs
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.gui.control import UserFeedbackManager, ufm
from holoviews.streams import SingleTap, DoubleTap
import geoviews.feature as gf
import panel as pn
from panel.layout import Panel
import xarray as xa, numpy as np
import os, glob
from enum import Enum
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.gui.spatial.widgets.markers import Marker

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

def crange( data: xa.DataArray, idim:int ) -> str:
    sdim = data.dims[idim]
    c: np.ndarray = data.coords[sdim].values
    return f"[{c.min():.2f}, {c.max():.2f}]"

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

#BoundsStream = Stream.define('bounds', bounds=param.Tuple(default=(0.0,0.0,0.0,0.0), doc='Image Bounds') )

class RGBViewer(param.Parameterized):
    rgb = param.Tuple( default=(50,150,300))
    bounds = param.Tuple( default=(0.0,0.0,0.0,0.0) )

    def __init__(self, **plotopts):
        super(RGBViewer, self).__init__()
        self.width = plotopts.get('width', 600)
        self.height = plotopts.get('height', 500)
        self._global_bounds = None

    @property
    def bands(self) -> np.ndarray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return tm().tile.data.band.values

    def init_gui(self,**kwargs):
        bmax = int(self.bands[-1])
        self.tap_stream = SingleTap( transient=True )
        self.selection_dmap = hv.DynamicMap( self.select_points, streams=[self.tap_stream] )
        self.point_graph = hv.DynamicMap( self.update_graph, streams=[self.tap_stream] )
        self.rplayer: IntSlider = IntSlider( name='Red',   start=0, end=bmax, value=self.rgb[0] )
        self.gplayer: IntSlider = IntSlider( name='Green', start=0, end=bmax, value=self.rgb[1] )
        self.bplayer: IntSlider = IntSlider( name='Blue',  start=0, end=bmax, value=self.rgb[2] )
        self.image = hv.DynamicMap( self.get_image, streams=dict( br=self.rplayer.param.value,
                                                                  bg=self.gplayer.param.value,
                                                                  bb=self.bplayer.param.value,
                                                                  bounds=self.param.bounds ) )
        self._global_image: hv.RGB = None
        self.band_markers = hv.DynamicMap( self.get_band_markers, streams=dict( br=self.rplayer.param.value,
                                                                                bg=self.gplayer.param.value,
                                                                                bb=self.bplayer.param.value ) )
    @property
    def global_image(self) -> hv.RGB:
        if self._global_image is None:
            self._global_image = self.get_global_image()
        return self._global_image

    def get_band_markers(self, br: int, bg: int, bb: int ) -> hv.Overlay:
        rm = hv.VLine(br).opts(color="red")
        gm = hv.VLine(bg).opts(color="green")
        bm = hv.VLine(bb).opts(color="blue")
        return hv.Overlay( [rm,gm,bm] )

    @exception_handled
    def select_points(self, x, y):
        result = hv.Points([(x,y)]).opts(marker='+', size=12, line_width=3, angle=45, color='white')
        return result

    def band_norm(self, data: xa.DataArray ) -> xa.DataArray:
        ndata = data / np.max( np.nan_to_num( data.values, nan=0 ) )
        return data.copy( data=ndata )

    def set_image_bounds(self, block: Block ):
        bounds: Tuple[float, float, float, float] = block.bounds()
        lgm().log( f"#RGB: set_image_bounds={bounds}")
        self.bounds = bounds

    @exception_handled
    def update_graph(self, x, y) -> hv.Curve:
        from spectraclass.learn.pytorch.trainer import stat
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        graph_data: xa.DataArray = self.band_norm( tm().tile.data.sel(x=x, y=y, method="nearest") )
        lgm().log( f"#RGB: Plotting graph_data[{graph_data.dims}]: shape = {graph_data.shape}, dims={graph_data.dims}, stat={stat(graph_data)}")
        popts = dict( width=self.width, height=200, yaxis="bare", ylim=(0,1.0) )
        data_table: hv.Table = hv.Table((graph_data.band.values, graph_data.values), 'Band', 'Spectral Value')
        current_curve = hv.Curve(data_table).opts(line_width=1, line_color="black", **popts)
        return current_curve

    def get_data(self, br: int, bg:int, bb: int ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return tm().tile.rgb_data( (br,bg,bb) )

    @exception_handled
    def get_image(self, br: int, bg:int, bb: int, bounds: Tuple[float,float,float,float]=(0.0,0.0,0.0,0.0) ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        RGB: xa.DataArray = self.get_data(br, bg, bb)
        if (bounds[0] == 0.0) and (bounds[1] == 0.0):
            x: np.ndarray = RGB.coords['x'].values
            y: np.ndarrayy = RGB.coords['y'].values
            dx, dy = (x[1]-x[0])/2, (y[1]-y[0])/2
            extent = ( x[0]-dx, y[0]-dy, x[-1]+dx, y[-1]+dy )
            self._global_bounds = extent
            block_image = RGB.values
            lgm().log( f"#RGB({br},{bg},{bb}): RGB.shape={RGB.shape}, nbands={tm().tile.data.shape[0]}, xlen={x.size}, ylen={y.size}, bounds={extent}" )
        else:
            (x0, y0, x1, y1) = bounds
            extent =( x0, y1, x1, y0)
            lgm().log( f"#RGB({br},{bg},{bb}): RGB.shape={RGB.shape}, nbands={tm().tile.data.shape[0]}, bounds={extent}")
            block_image = self.subset_data( RGB, extent ).values
        return hv.RGB( block_image, bounds=extent ).opts( width=self.width, height=self.height, xlim=(extent[0],extent[2]), ylim=(extent[3],extent[1]) )

    @exception_handled
    def get_global_image(self) -> hv.RGB:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        RGB: xa.DataArray = tm().tile.rgb_data( self.rgb )
        extent =  tm().extent
        lgm().log( f"#RGB: get_global_image, shape={RGB.shape}, extent={extent}" )
        return hv.RGB( RGB.values.copy(), bounds=extent ).opts( width=self.width, height=self.height,
                        xlim=(extent[0],extent[2]), ylim=(extent[3],extent[1]), shared_datasource=False )

    def subset_data(self, data: xa.DataArray, bounds: Tuple[float,float,float,float] ) -> xa.DataArray:
        return data.sel(x=slice(bounds[0],bounds[2]), y=slice(bounds[1],bounds[3]))

    def panel(self,**kwargs):
        self.init_gui(**kwargs)
        block_image = pn.Column( self.image*self.selection_dmap, self.point_graph*self.band_markers, self.rplayer, self.gplayer, self.bplayer )
        return pn.Tabs( ('block',block_image), ('tile', self.global_image) )

class VariableBrowser:

    def __init__(self, cname: str, **plotopts ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        self.cname = cname
        self._block_selection: int = None
        lgm().log( f"Creating VariableBrowser {cname}", print=True)
        self.data: xa.DataArray = sgui().get_data(cname)
        lgm().log(f" --> data shape = {self.data.shape}", print=True)
        self.width = plotopts.get('width', 600)
        self.height = plotopts.get('height', 500)
        self.cmap = plotopts.get('cmap', 'jet')
        self._point_selection_enabled = False
        self.nIter: int = self.data.shape[0]
        self.player: pn.widgets.DiscretePlayer = pn.widgets.DiscretePlayer(name='Iteration', options=list(range(self.nIter)), value=self.nIter - 1)
        self.tap_stream = SingleTap( transient=True )
        self.selection_dmap = hv.DynamicMap( self.select_points, streams=[self.tap_stream] )
        self.point_graph = hv.DynamicMap( self.update_graph, streams=[self.tap_stream] )
        self.image = hv.DynamicMap(self.get_frame, streams=dict(iteration=self.player.param.value, block_selection=tm().block_selection.param.index))
        self.iter_marker = hv.DynamicMap( self.get_iter_marker, streams=dict( index=self.player.param.value ) )
        self.graph_data = xa.DataArray([])
        self.curves: List[hv.Curve] = []
        self.current_curve_data: Tuple[int,hv.Curve] = None

    def point_selection_enabled(self,  enabled: bool ):
        self._point_selection_enabled = enabled

    def update_point_selection(self, activate: bool ):
        if activate: self.tap_stream.add_subscriber( self.point_graph )
        else: self.tap_stream.clear()

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
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        ts = time.time()
        block: Block = tm().getBlock()
        graph_data: xa.DataArray = block.filtered_raster_data.sel(x=x, y=y, method="nearest")
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
        data_table: hv.Table = hv.Table((graph_data.band.values, graph_data.values), 'Band', 'Spectral Value')
        current_curve = hv.Curve(data_table).opts(line_width=3, line_color = line_color, **popts)
        self.current_curve_data = ( lm().current_cid, current_curve )
        new_curves = [ current_curve ]
        t1 = time.time()
        if is_probe:
            reproduction: xa.DataArray = block.getReproduction(raster=True)
            verification_data: xa.DataArray = reproduction.sel( x=x, y=y, method="nearest" )
            lgm().log(f"V%%  [{self.cname}]  input_data       shape={graph_data.shape}, dims={graph_data.dims}, "
                      f"range={crange(graph_data,0)}, vange={arange(graph_data)}")
            lgm().log( f"V%% [{self.cname}] verification_data shape={graph_data.shape}, dims={verification_data.dims}, "
                       f"range={crange(verification_data,0)}, vange={arange(verification_data)}" )
            verification_table: hv.Table = hv.Table((verification_data.band.values, verification_data.values), 'Band', 'Spectral Value')
            verification_curve = hv.Curve( verification_table ).opts( line_width=1, line_color='grey', **popts )
            new_curves.append( verification_curve )
        t2 = time.time()
        updated_curves = self.curves + new_curves
        result =  hv.Overlay( updated_curves )
        tf = time.time()
        lgm().log( f"TT: update_graph dt={tf-ts} t1={t1-ts} t2={t2-ts}")
        return result

    def update_block(self, block_selection: int ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if self._block_selection != block_selection:
            lgm().log(f" ------------>>>  VariableBrowser[{self.cname}].select_block: {block_selection}  <<<------------ ")
            bcoords = tm().bi2c(block_selection)
            print( f"Select block {bcoords}")
            self.data = sgui().get_data( self.cname, bindex=bcoords )
            self._block_selection = block_selection

    @exception_handled
    def get_frame(self, iteration: int, block_selection: int ) -> hv.Image:
        from spectraclass.learn.pytorch.trainer import stat
        ts = time.time()
        if block_selection >= 0:
            lgm().log( f"#VB: {self.cname}-> get_frame: iteration={iteration} block_selection={block_selection} ")
            self.update_block( block_selection )
        fdata: xa.DataArray = self.data[iteration]
        lgm().log(f"#VB: {self.cname}-> get_frame: raw data dims={self.data.dims} shape={self.data.shape} stat={stat(self.data)};  fdata dims={fdata.dims} shape={fdata.shape} stat={stat(fdata)}")
        xlim, ylim = bounds( fdata )
        iopts = dict(width=self.width, height=self.height, cmap=self.cmap, clim=self.data.attrs['clim'], xaxis="bare",
                     yaxis="bare", x="x", y="y", colorbar=False, xlim=xlim, ylim=ylim )
        t2 = time.time()
        result: hv.Image = fdata.hvplot.image( **iopts )
        lgm().log(f"#VB: iteration={iteration}, block={block_selection}, data shape={fdata.shape}, stat={stat(fdata)}, result: {result}")
        tf = time.time()
        lgm().log( f"#VB: get_frame dt={tf-ts} t2={t2-ts}")
        return result

    @exception_handled
    def get_iter_marker(self, index: int ):
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
            image_panel = self.image * self.selection_dmap  # * rs().get_selector()
            selector = lm().class_selector
            return pn.Column( selector, image_panel, self.player, self.point_graph*self.iter_marker )
        else:
            return pn.Column( self.image, self.player )

class hvSpectraclassGui(SCSingletonConfigurable):

    def __init__(self):
        super(hvSpectraclassGui, self).__init__()
        self.browsers: Dict[str,VariableBrowser] = None
        self.panels: List = None
        self.color_range = 2.0
        self.mapviews: pn.Tabs = None
        self.alert = ufm().gui()

    @exception_handled
    def init( self, **plotopts ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        collections = [ "bands", 'features', "reproduction" ]
        self.browsers = { cname: VariableBrowser( cname ) for cname in collections }
        self.browsers['bands'].verification = plotopts.pop('verification',None)
        self.panels = [ (cname,browser.plot(**plotopts)) for cname, browser in self.browsers.items() ]
        self.panels.append(('satellite', tm().satellite_block_view) )
        self.panels.append(('rgb', tm().rgbviewer ))
        self.panels.append( ('clusters', clm().panel() ) )
        self.mapviews = pn.Tabs( *self.panels, dynamic=True )
        self.tab_watcher = self.mapviews.param.watch(self.on_tab_change, ['active'], onlychanged=True)
        return self

    def learning_test( self, rpolys: List[Dict[str,Union[np.ndarray,int]]], **kwargs ):
        from spectraclass.learn.pytorch.trainer import mpt
        self.add_test_markers( rpolys, **kwargs )
        mpt().train()

    def add_test_markers(self, rpolys: List[Dict[str,Union[np.ndarray,int]]], **kwargs ) -> List[Marker]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        markers = []
        for rpoly in rpolys:
            marker = tm().get_region_marker( rpoly, rpoly['ic'], **kwargs )
            markers.append( marker )
            lm().addMarker(marker)
        return markers

    @exception_handled
    def on_tab_change(self, *events):
        for event in events:
            lgm().log(f"on_tab_change: {event.old} -> {event.new}" )
 #           new_panel = self.mapviews.objects[ event.new ]
 #           old_panel = self.mapviews.objects[ event.old ]
            # if hasattr( new_panel, 'objects' ) and hasattr( old_panel, 'objects' ):
            #     new_slider: DiscretePlayer = new_panel.objects[2]
            #     old_slider: DiscretePlayer = old_panel.objects[2]
            #     if len(new_slider.values) == len(old_slider.values):
            #         new_slider.value.apply.opts( value=old_slider.value )

    def get_data( self, cname: str, **kwargs ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.base import dm
        sfactor = kwargs.get('sfactor', 2.0)
        block: Block = tm().getBlock( **kwargs )
        lgm().log( f"sgui:get_data[{cname}] block = {block.index}")
        if cname=="bands":
            result =  tm().prepare_inputs(block=block, raster=True, **kwargs)
            result.attrs['clim'] = (-self.color_range,self.color_range)
        elif cname=="features":
            result = dm().getModelData(block=block, raster=True, norm=True)
            result.attrs['clim'] = self.get_clim( result.values, sfactor )
        elif cname=="reproduction":
            result = block.getReproduction(raster=True)
            result.attrs['clim'] = (-self.color_range,self.color_range)
        else:
            raise Exception( f"Unkonwn data type: {cname}")
        return result

    def get_clim(self, data: np.ndarray, sfactor: float) -> Tuple[float,float]:
        dmean, drng = data.mean(), sfactor*data.std()
        return ( dmean-drng, dmean+drng )

    @exception_handled
    def get_control_panel(self,**kwargs) -> Panel:
        from spectraclass.data.base import dm
        from spectraclass.learn.cluster.manager import clm
    #    from spectraclass.gui.pointcloud import PointCloudManager, pcm
        data_selection_panel = pn.Tabs(  ("Tile", dm().modal.get_tile_selection_gui(**kwargs)) ) # , ("Block",dm().modal.gui()) ] )
   #     manifold_panel = pn.Row( pcm().gui() )
        analytics_gui = pn.Tabs( ("Cluster", clm().gui()), ("Classify", rs().get_control_panel() ), ("Mask", mm().get_control_panel() ) )
        controls = pn.Accordion( ('Data Selection', data_selection_panel ), ('Analytics',analytics_gui), toggle=True, active=[0] ) # , ('Manifold', manifold_panel )
        return pn.Column( self.alert, controls )

    @exception_handled
    def panel(self, title: str = None, **kwargs ) -> Panel:
        rows = [ self.mapviews ]
        if title is not None: rows.insert( 0, title )
        image_column = pn.Column( *rows )
        return pn.Row(  image_column, self.get_control_panel(**kwargs) )

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
