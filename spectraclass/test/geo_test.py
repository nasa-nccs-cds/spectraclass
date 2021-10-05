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
gv.extension('bokeh')
import holoviews as hv
from holoviews import opts

SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
data_array: xa.DataArray = rio.open_rasterio( SpectralDataFile, chunks=True )
print( data_array.spatial_ref )
# desis_image: DynamicMap = data_array.hvplot.image( cmap='jet', tiles="EsriImagery", crs=data_array.spatial_ref.crs_wkt )
