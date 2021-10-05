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
from holoviews.core.spaces import DynamicMap
from spectraclass.gui.spatial.widgets.tiles import TileSelector, TileManager
import cartopy.crs as ccrs
gv.extension('bokeh')
import holoviews as hv
from holoviews import opts
iband = 0
block_size = 100
origin = "upper"
blocks_per_tile = 5
tile_size = block_size*blocks_per_tile

SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"

tmgr = TileManager( SpectralDataFile, tile_size, origin, nodata_fill=-1 )
data_array: xa.DataArray = TileManager.read_data_layer( SpectralDataFile, origin )
crs: ccrs.CRS = tmgr.crs()

# desis_image: DynamicMap = data_array.hvplot.image( cmap='jet', tiles="EsriImagery", crs=data_array.spatial_ref.crs_wkt )
