import traceback, linecache
from typing import List, Union, Tuple, Optional, Dict, Type, Hashable, Callable
import hvplot.pandas
import hvplot.xarray
import geoviews as gv
import holoviews as hv
import pandas as pd
from panel.layout import Panel
import geoviews.feature as gf
import panel as pn
from panel.layout import Panel
import xarray
import xarray as xa, numpy as np
import os, glob
from enum import Enum
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

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

    def __init__(self, data: xa.DataArray ):
        self.data = data

    def plot(self, **plotopts)-> Panel:
        width = plotopts.get('width',600)
        widget_type = plotopts.get('widget_type','scrubber')
        cmap = plotopts.get('cmap', 'jet')
        iopts = dict( width=width, cmap=cmap, xaxis = "bare", yaxis = "bare"  )
        return  self.data.hvplot.image( groupby='band', widget_type=widget_type, widget_location='bottom', **iopts )

class RasterCollectionsViewer:

    def __init__(self, collections: Dict[str,xa.DataArray], **plotopts ):
        self.browsers = { cname: VariableBrowser( cdata ) for cname,cdata in collections.items() }
        self.panels = [ (cname,browser.plot(**plotopts)) for cname,browser in self.browsers.items() ]

    def panel(self, title: str = None, **kwargs ) -> Panel:
        items = [ pn.Tabs( *self.panels ) ]
        if title is not None: items.insert(0,title)
        background = kwargs.get( 'background', 'WhiteSmoke')
        return pn.Column( *items, background=background )
