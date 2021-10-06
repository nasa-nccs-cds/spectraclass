#%%

import numpy as np
import xarray as xa
import hvplot.xarray
import panel as pn
import rioxarray as rio
import pandas as pd
import geoviews as gv
import geoviews.feature as gf
import geoviews.tile_sources as gts
from geoviews.element.geo import WMTS
from holoviews.plotting.links import DataLink
from spectraclass.gui.spatial.widgets.tiles import TileSelector, TileManager
import cartopy.crs as ccrs
from holoviews.core.spaces import DynamicMap
gv.extension('bokeh')
import holoviews as hv
from holoviews import opts
from bokeh.layouts import column
from bokeh.models import Slider

#%%

# tilemap: WMTS = gts.EsriImagery.opts(width=400, height=400 )

iband = 0
block_size = 100
plot_size = [ 500, 500 ]
origin = "upper"
blocks_per_tile = 5
tile_size = block_size*blocks_per_tile

SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"

tmgr = TileManager( SpectralDataFile, tile_size, origin, nodata_fill=-1 )
data_array: xa.DataArray = TileManager.read_data_layer( SpectralDataFile, origin )
crs: ccrs.CRS = tmgr.crs()

slider = Slider(start=0.0, end=1.0, step=0.01, value=1.0)
desis_image: DynamicMap = data_array.hvplot.image( cmap='jet', tiles="EsriImagery", width=plot_size[0], height=plot_size[1], crs=crs )
fig = hv.render(desis_image)
slider.js_link('value', fig.renderers[-1].glyph, 'global_alpha' )
