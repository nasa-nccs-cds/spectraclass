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

class Layer(param.Parameterized):
    alpha = param.Magnitude()
    visible = param.Boolean(True)

    def __init__(self, name: str ):
        param.Parameterized.__init__(self)
        self.name = name

    def panel(self):
        return pn.Param(self.param, parameters=[ 'alpha', 'visible'], name=self.name )
