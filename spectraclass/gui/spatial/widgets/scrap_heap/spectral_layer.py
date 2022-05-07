import param
import os, time, traceback
import holoviews as hv
import numpy as np
import xarray as xa
import hvplot.xarray
from holoviews.plotting.links import RangeToolLink
from bokeh.models.mappers import CategoricalColorMapper
from spectraclass.gui.spatial.widgets.points import PointSelection
import panel as pn
import geoviews as gv
from collections import OrderedDict
import geoviews.tile_sources as gts
from geoviews.element import WMTS
import rioxarray as rio
import rasterio
from holoviews import streams
from bokeh.io import push_notebook, show, output_notebook
from holoviews import opts
from typing import List, Dict, Tuple, Optional
from spectraclass.xext.xgeo import XGeo
import logging
hv.extension('bokeh')

# from bokeh.models.tools import BoxSelectTool
# from holoviews.plotting.bokeh.renderer import BokehRenderer
# bokeh_renderer = BokehRenderer.instance(mode='server')

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
log_file = os.path.expanduser('~/.spectraclass/logging/geospatial.log')
file_handler = logging.FileHandler(filename=log_file, mode='w')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

def exception_handled(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            logger.error(f" Error in {func}:")
            logger.error(traceback.format_exc())

    return wrapper

def print_bokeh_attrs(hv_instance):
    bokeh_model = hv.render(hv_instance)
    bokeh_attrs = bokeh_model.properties_with_values(include_defaults=True)
    print(f" {hv_instance.__class__} attrs:")
    for (key, value) in bokeh_attrs.items(): print(f"{key}: {value}")
    return bokeh_model

class SpectralLayer(param.Parameterized):
    band = param.Integer(default=0)
    alpha = param.Magnitude()
    color_range = param.Range()
    class_selector = param.ObjectSelector( objects=[] )
    cmap = param.ObjectSelector( objects=hv.plotting.util.list_cmaps(), default="jet" )
    bands_visible = param.Boolean(True)
    basemap_visible = param.Boolean(True)
    rescale_colors = param.Boolean(False)
    classify_selection = param.Boolean(False)
    default_plot_args = dict(width=500, height=500)

    def __init__(self, raster: xa.DataArray, **kwargs):
        param.Parameterized.__init__(self)
        self.raster = raster
        self._maxval = np.nanmax( self.raster.values )
        self._class_selections = {}
        self._class_map = OrderedDict()
        self._class_map['unclassified'] = 'white'
        self._class_map.update( kwargs.pop('classes', {} ) )
        self._class_list = list(self._class_map.keys())
        self._class_colors = list(self._class_map.values())
        self.points = hv.Points(([], [], []), vdims='color')
        self.point_size = kwargs.get('size', 5)
        self.point_stream = streams.PointDraw(source=self.points, empty_value='white')
        self.polys = hv.Polygons([])
        self.poly_draw_stream = streams.PolyDraw(source=self.polys, drag=False, show_vertices=True, styles={ 'fill_color': self._class_colors[1:] })
        self.poly_edit_stream = streams.PolyEdit(source=self.polys, vertex_style={'color': 'red'}, shared=True)
        self.tap_stream = streams.SingleTap( transient=True )
        self.double_tap_stream = streams.DoubleTap( rename={'x': 'x2', 'y': 'y2'}, transient=True)
        self.bounds = self.raster.xgeo.bounds()
        self.param.class_selector.objects = self._class_list
        self.class_selector = self.param.class_selector.default = self._class_list[0]
        self.range_stream = streams.RangeXY()
        self._tile_source = None
        self._raster_range = (float(self.raster.min(skipna=True)), float(self.raster.max(skipna=True)))
        self._current_band = -1
        self.tools = ['pan', 'box_zoom', 'wheel_zoom', 'hover', 'undo', 'redo', 'reset']
        self._image: hv.Image = None
        self._plot_args = dict(**self.default_plot_args)
        self._plot_args.update(**kwargs)
        self._color_range = (raster.values.min(), raster.values.max())
        self._count = -1
        self._graph = None

    def get_fill_color(self):
        self._count = self._count + 1
        color = self._colors[ self._count % len(self._colors) ]
        logger.info( f"get_fill_color: {color}" )
        return color

    @property
    def decreasing_y(self):
        return ( self.raster.y[0] > self.raster.y[-1] )

    @exception_handled
    def control_panel(self):
        panels = [self._get_map_panel()]
        class_list = list(self._class_map.keys())
        if len( class_list ) > 0:
            class_panel = pn.Param( self.param, parameters=['class_selector','classify_selection'], name="classes",
                                    widgets={'classify_selection': {'widget_type': pn.widgets.Button}})
            panels.append(class_panel)
        return pn.Tabs( *panels )

    def _get_map_panel(self):
        rng, shp = self._raster_range, self.raster.shape
        map_panel = pn.Param(self.param, name="map",
                             parameters=['band', 'cmap', 'color_range', 'rescale_colors', 'alpha', 'bands_visible', 'basemap_visible' ],
                             widgets={'band': {'widget_type': pn.widgets.IntSlider, 'start': 0, 'end': shp[0] - 1},
                                      'color_range': {'widget_type': pn.widgets.RangeSlider, 'start': rng[0], 'end': rng[1]},
                                      'rescale_colors': {'widget_type': pn.widgets.Button}})
        return map_panel

    @exception_handled
    def get_basemap(self, basemap: str = "ESRI", **kwargs) -> hv.Image:
        if self._tile_source is None:
            if basemap.upper() == "ESRI": basemap = "EsriImagery"
            [xmin, ymin, xmax, ymax] = self.bounds
            self._tile_source = gts.tile_sources.get(basemap, None).opts(xlim=(xmin, xmax), ylim=(ymin, ymax), **kwargs)
        return self._tile_source

    @exception_handled
    def get_coastline(self, scale='10m', **kwargs) -> Optional[hv.Image]:
        coastline = gv.feature.coastline()
        assert scale in ['10m', '50m',
                         '110m'], f"Unrecognized coastline scale: {scale}, must be one of '10m', '50m' or '110m'"
        return coastline.opts(scale=scale, **kwargs)

    def filter_cmaps(self, provider=None, records=False, name=None, category=None, source=None, bg=None, reverse=None):
        self.cmap.objects = hv.plotting.util.list_cmaps(provider, records, name, category, source, bg, reverse)

    @property
    def layer(self) -> xa.DataArray:
        return self.raster[self.band].squeeze(drop=True)

    def get_image(self):
        if (self._image is None) or (self.band != self._current_band):
            self._image = hv.Image(self.layer)
            self.range_stream.source = self._image
        return self._image

    @exception_handled
    def update_clim(self, **kwargs):
        xy_ranges = [kwargs.get('x_range', None), kwargs.get('y_range', None)]
        [xr, yr] = [[] if r is None else list(r) for r in xy_ranges]
        logger.info( f" update_clim, xr={xr}, yr={yr}, rescale_colors={self.rescale_colors}")
        if self.rescale_colors:
            if self.decreasing_y: (yr[1], yr[0]) = (yr[0], yr[1])
            subimage = self.layer.loc[yr[0]:yr[1], xr[0]:xr[1]]
            self._color_range = (float(subimage.min(skipna=True)), float(subimage.max(skipna=True)))
            self.rescale_colors = False
            push_notebook()
        elif self.color_range:
            self._color_range = self.color_range

    @exception_handled
    def image(self, **kwargs):
        self.update_clim(**kwargs)
        current_image = self.get_image()
        image = current_image.opts(cmap=self.cmap, alpha=self.alpha, clim=self._color_range, visible=self.bands_visible,
                                   tools=self.tools, **self._plot_args)
        self._current_band = self.band
        self.tap_stream.source = image
        self.double_tap_stream.source = image
        return image

    @exception_handled
    def graph_selection(self, **kwargs ) -> hv.Curve:
        rescale = False
        x,y,x2,y2,z = kwargs.get('x'), kwargs.get('y'), kwargs.get('x2'), kwargs.get('y2'), None
        if x is not None:
            z = self.raster.sel( x=x, y=y, method="nearest" ).squeeze()
        elif x2 is not None:
            z = self.raster.sel( x=x2, y=y2, method="nearest" ).squeeze()
            rescale = True
        elif self._graph is None:
            z = self.raster.isel( x=0, y=0 ).squeeze()
        if z is not None:
            graph_data = hv.Dataset( z )
            range = (np.nanmin(z.values), np.nanmax(z.values)) if rescale else (0.0, self._maxval)
            self._graph = graph_data.to( hv.Curve, kdims=[self.raster.dims[0]] )\
                .relabel( 'Point Spectra' )\
                .redim( z=hv.Dimension( "spectra", label='Spectral Intensity', range=range ) )
        return self._graph

    @param.depends( 'band', 'alpha', 'cmap', 'bands_visible', 'rescale_colors', 'color_range', 'class_selector', 'basemap_visible' )
    def dmap_spectral_plot(self, **kwargs) -> hv.Overlay:
        logger.info(f"dmap_spectral_plot, args: {kwargs}")
        self.graph_selected_elements( **kwargs )
        image = self.image(**kwargs)
#        class_selections = self.process_selection( **kwargs )
        layers = [ self.get_basemap() ] if self.basemap_visible else []
        layers.append( image )
        return hv.Overlay( layers )

    def dmap_graph(self, **kwargs):
        graph = self.graph_selection(**kwargs)
        return graph.opts( width=800, height=250 )

    # def clear_temp_polys(self):
    #     self._poly_temp = self._init_poly_temp
    #     for stream in [ self.poly_draw_stream, self.poly_edit_stream ]:
    #         stream.reset()
    #         stream.source = self._poly_temp

    # @exception_handled
    # def process_selection( self, **kwargs ) -> hv.Polygons:
    #     polys = hv.Polygons([])
    #     if self.classify_selection:
    #         self.classify_selection = False      # testl;mlag
    #         selections = kwargs.get('edited_data', kwargs.get('data', []))
    #         if len(selections):
    #             class_color = self._class_map[self.class_selector]
    #             iC = self._class_colors.index( class_color )
    #             xs: List[List[float]] = selections.get( 'xs', [] )
    #             ys: List[List[float]] = selections.get( 'ys', [] )
    #             for (x,y) in zip(xs,ys):
    #                 key = ( x[0], y[0] )
    #                 if key not in self._class_selections:
    #                     self._class_selections[ key ] = {'x': x, 'y': y, 'color': class_color }
    #                     logger.info( f"\n\nADD class selection, class: {self.class_selector}, color: {class_color}")
    #             pdata = list( self._class_selections.values() )
    #             logger.info(f" ---> SELECTION pdata: {pdata}")
    #             polys = hv.Polygons( pdata, vdims='color' ).opts( color='color', line_width=1  ) # , cmap=self._class_colors )
    #             self.clear_temp_polys()
    #     return polys * self._poly_temp

    @exception_handled
    def graph_selected_elements(self, **kwargs):
        data = kwargs.get('edited_data', kwargs.get('data'))
        if data is not None:
            logger.info(f" raster dims: {self.layer.dims}, shape: {self.layer.shape}, coords: {self.layer.coords}")
            geometries = []
            boundaries = []
            for (ys, xs) in zip(data['ys'], data['xs']):
                coords = list(zip(ys, xs))
                logger.info(f" coords: {coords}")
                geometries.append(dict(type='Polygon', coordinates=[coords]))
                boundaries.append(coords)
#           regions = regionmask.Regions( boundaries )
#           regions.mask()
            clipped = self.layer.rio.clip(geometries, crs=self.raster.crs)
            logger.info(f" clipped: {clipped}")

#     @param.depends( 'class_selector' )
#     def get_selection(self, **kwargs ):
#         try:
#             class_color = self._class_map[ self.class_selector ]
# #            fill_colors = self.poly_draw_stream.styles['fill_color']
# #            fill_colors.append( class_color )
#             logger.info(f"Setting poly fill color: {class_color}")
#             self.poly_draw_stream.styles['fill_color'] = [ class_color ]
#             self.poly_draw_stream.styles['fill_alpha'] = 1.0
#         except Exception:
#             logger.error( traceback.format_exc() )
#         return self.polys.opts( opts.Polygons( active_tools=self.poly_streams ) )

    # @param.depends('class_selector', watch=True)
    # def update_selection(self):
    #     try:
    #         class_color = self._class_map[self.class_selector]
    #         logger.info(f"Setting poly fill color: {class_color}")
    #         self.poly_draw_stream.styles.update( fill_color = [ class_color ]  )
    #     except Exception:
    #         logger.error(traceback.format_exc())

    def process_points(self, element ):
        logger.info( f" ** process_points: {element}" )
        return element

    @exception_handled
    def graph(self):
        streams = [ self.tap_stream, self.double_tap_stream ]
        graph = hv.DynamicMap( self.dmap_graph, streams=streams  )
        return graph

    @exception_handled
    def plot(self):
        image = hv.DynamicMap( self.dmap_spectral_plot, streams=[ self.range_stream, self.point_stream ] )
        polys = self.polys.opts( opts.Polygons( fill_alpha=0.8, active_tools=['poly_draw'] ))
        return image * polys



