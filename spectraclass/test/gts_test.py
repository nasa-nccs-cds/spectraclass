import geoviews.tile_sources as gts
import xarray as xa
import holoviews as hv
import panel as pn
from typing import List, Union, Tuple, Optional, Dict, Callable
pn.extension()
hv.extension('bokeh')

plot = gts.EsriImagery.opts(width=600, height=570, global_extent=True)
plot