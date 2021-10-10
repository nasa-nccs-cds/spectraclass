import os, time
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
origin = "upper"
SpectralDataFile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
result_dir = "/Users/tpmaxwel/Development/Data/raster/tiles"
crs = "+a=6378137.0 +b=6378137.0 +nadgrids=@null +proj=merc +lon_0=0.0 +x_0=0.0 +y_0=0.0 +units=m +no_defs"

t0 = time.time()
base_name = os.path.basename(os.path.splitext(SpectralDataFile)[0])
result_path = os.path.join( result_dir, f"{base_name}.nc" )
os.makedirs( result_dir, exist_ok=True )
data: xa.DataArray = xa.open_rasterio( SpectralDataFile )
raster: xa.DataArray = TileManager.to_standard_form( data, origin, ccrs=False )
# chunks = { raster.dims[0]: 1, raster.dims[1]: raster.shape[1], raster.dims[2]: raster.shape[2] }
# print( f"Loading file '{SpectralDataFile}' with shape: {raster.shape}, chunks: {chunks}")
# dset = xa.Dataset( dict( z=raster ), attrs=raster.attrs ).chunk( chunks )

if crs:
    print( f"Reprojecting to '{crs}'" )
    raster = raster.rio.reproject( crs )

raster.attrs['crs'] = crs
dset = xa.Dataset( dict( z=raster ), attrs=raster.attrs )
dset.to_netcdf( result_path, mode="w" )

print( f"Completed generating file '{result_path}' in total time = {time.time()-t0} sec.")


