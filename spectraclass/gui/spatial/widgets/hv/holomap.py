#%%

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
from spectraclass.gui.spatial.widgets.tiles import TileSelector, TileManager
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
hv.extension('bokeh')

class HPlot(param.Parameterized):

    def __init__(self, raster: xa.DataArray, **kwargs ):
        param.Parameterized.__init__(self)
        self.raster = raster
        self.dset = hv.Dataset(raster)
        self._tile_source = None
        self.box_select: BoxSelectTool = BoxSelectTool()
        self._color_range = ( raster.values.min(), raster.values.max() )
        self.tools = [ 'hover' ]
        rdims = list( self.raster.dims )
        self._image  = self.dset.to( hv.Image, kdims=[rdims[-2],rdims[-1]], dynamic=kwargs.pop('dynamic',True) )
        self._opts = dict( tools = self.tools, clim=self._color_range, width=500, height=500 )
        self._opts.update( **kwargs )

    def get_basemap(self, basemap: str = "ESRI", **kwargs ) -> hv.Image:
        if self._tile_source is None:
            if basemap.upper() == "ESRI": basemap = "EsriImagery"
            [ xmin, ymin, xmax, ymax ] = self.raster.xgeo.bounds()
            self._tile_source =  gts.tile_sources.get( basemap, None ).opts( xlim=(xmin,xmax), ylim=(ymin,ymax), **kwargs )
        return self._tile_source

    def get_coastline( self, scale='10m', **kwargs ) -> Optional[hv.Image]:
        coastline = gv.feature.coastline()
        assert scale in ['10m', '50m', '110m'], f"Unrecognized coastline scale: {scale}, must be one of '10m', '50m' or '110m'"
        return coastline.opts( scale=scale, **kwargs )

    def plot(self, **kwargs ):
        opts = dict( **self._opts )
        opts.update( **kwargs )
        return self._image.opts( **opts )
