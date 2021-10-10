import os, time
import numpy as np
import xarray as xa
import hvplot.xarray
import panel as pn
import rioxarray as rio
import pandas as pd
import geoviews as gv
from spectraclass.xext.xgeo import XGeo
import geoviews.feature as gf
import geoviews.tile_sources as gts
from geoviews.element.geo import WMTS
from holoviews.element import Dataset as hvDataset
from holoviews.plotting.links import DataLink
from spectraclass.gui.spatial.widgets.tiles import TileSelector, TileManager
from holoviews.core import Dimension
import cartopy.crs as ccrs
from holoviews.core.spaces import DynamicMap
gv.extension('bokeh')
import holoviews as hv
from holoviews import opts
from bokeh.layouts import column
from hvplot.plotting.core import hvPlot

iband = 0
cdim = "band"
SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
result_dir = "/Users/tpmaxwel/Development/Data/raster/tiles"
crs = "+a=6378137.0 +b=6378137.0 +nadgrids=@null +proj=merc +lon_0=0.0 +x_0=0.0 +y_0=0.0 +units=m +no_defs"

t0 = time.time()
base_name = os.path.basename( os.path.splitext(SpectralDataFile)[0] )
result_path = os.path.join( result_dir, f"{base_name}.nc" )
os.makedirs( result_dir, exist_ok=True )
raster: xa.DataArray = rio.open_rasterio( SpectralDataFile, default_name='z' )
if crs: raster = raster.rio.reproject( crs )
# raster = raster.chunk( chunks={ cdim: 1 } )
raster.to_netcdf( result_path )

print( f"Completed generating file '{result_path}' in total time = {time.time()-t0} sec.")


# raster: xa.DataArray = TileManager.to_standard_form( data, origin, ccrs=False, name='z' )
# raster = raster.chunk( { raster.dims[0]: 1 } )


