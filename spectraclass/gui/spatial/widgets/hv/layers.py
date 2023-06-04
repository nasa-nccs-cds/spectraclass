import param
import os, time, traceback
import numpy as np
import xarray as xa
import hvplot.xarray
from holoviews.plotting.links import RangeToolLink
import panel as pn
import geoviews as gv
import geoviews.tile_sources as gts
from geoviews.element import WMTS
import rioxarray as rio
import rasterio
from holoviews import streams
import pandas as pd
from holoviews.element import Dataset as hvDataset
from holoviews.core.boundingregion import BoundingRegion, BoundingBox
from holoviews.plotting.links import DataLink
from holoviews.core import Dimension
import cartopy.crs as ccrs
from holoviews.core.spaces import DynamicMap
from bokeh.events import RangesUpdate, Pan
from holoviews.streams import RangeXY
import holoviews as hv
from bokeh.io import push_notebook, show, output_notebook
from holoviews import opts
from bokeh.layouts import column
from hvplot.plotting.core import hvPlot
from typing import List, Dict, Tuple, Optional
from spectraclass.xext.xgeo import XGeo
import logging
from bokeh.models.tools import BoxSelectTool

class Layer(param.Parameterized):
    alpha = param.Magnitude()
    visible = param.Boolean(True)
    default_layer_args = dict( width=500, height=500 )

    def __init__(self, name: str ):
        param.Parameterized.__init__(self)
        self.name = name

    def panel(self):
        return pn.Param(self.param, parameters=[ 'alpha', 'bands_visible'], name=self.name )


class ImageLayer( Layer ):
    band = param.Integer( default=0 )
    color_range = param.Range()
    cmap = param.Selector( objects=hv.plotting.util.list_cmaps(), default="jet" )
    rescale_colors = param.Boolean(False)

    def __init__(self, raster: xa.DataArray, **kwargs ):
        Layer.__init__(self)
        self.raster = raster
        self.bounds = self.raster.xgeo.bounds()
        self._raster_range = ( float(self.raster.min(skipna=True)), float(self.raster.max(skipna=True)) )
        self.tools = ['box_zoom', 'hover', 'box_select', 'lasso_select', 'poly_select', 'pan', 'wheel_zoom', 'tap', 'undo', 'redo', 'reset']
        self._current_band = -1
        self.range_stream: streams.RangeXY = streams.RangeXY( )
        self._image: hv.Image = None
        self._plot_args = dict( **self.default_layer_args )
        self._plot_args.update( **kwargs )
        self._color_range = ( float(raster.values.min()), float(raster.values.max()) )

    @property
    def decreasing_y(self):
        return ( self.raster.y[0] > self.raster.y[-1] )

    def colors_panel(self):
        color_range_specs = { 'widget_type': pn.widgets.RangeSlider, 'start': self._raster_range[0], 'end': self._raster_range[1]  }
        panel = pn.Param( self.param, parameters=[ 'cmap', 'color_range', 'rescale_colors'], name="colors",
             widgets={  'color_range': color_range_specs, 'rescale_colors': {'widget_type': pn.widgets.Button } } )
        return panel

    @property
    def layer(self) -> xa.DataArray:
        return self.raster[ self.band ]

    def get_image(self):
        if (self._image is None) or (self.band != self._current_band):
            self._image = hv.Image( self.layer )
            self.range_stream.source = self._image
        return self._image

    def update_clim( self, **kwargs ):
        try:
            yr = list(kwargs.get('y_range',[]))
            xr = list(kwargs.get('x_range',[]))
            if self.rescale_colors:
                if self.decreasing_y: (yr[1],yr[0]) = (yr[0],yr[1])
                subimage = self.layer.loc[ yr[0]:yr[1], xr[0]:xr[1] ]
                self._color_range = (float(subimage.min(skipna=True)), float(subimage.max(skipna=True)))
                self.color_range = self._color_range
                self.rescale_colors = False
                push_notebook()
            elif self.color_range:
                self._color_range =  self.color_range
        except Exception as err:
            print( f"ERROR: {err}")

    def plot( self, **kwargs ):
        self.update_clim( **kwargs )
        current_image = self.get_image()
        image = current_image.opts( cmap=self.cmap, alpha=self.alpha, clim=self._color_range, visible=self.visible, tools=self.tools, **self._plot_args )
        basemap = self.get_basemap( )
        self._current_band = self.band
        return ( image * basemap )
