import os, time
import numpy as np
import xarray as xa
import hvplot.xarray
import panel as pn
import rioxarray as rio
import rasterio
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
result_dir = "/Users/tpmaxwel/Development/Data/raster/tiles"
SpectralDataset = "/Users/tpmaxwel/Development/Data/raster/tiles/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.nc"
dataset: xa.Dataset = xa.open_dataset( SpectralDataset )
raster: xa.DataArray = dataset.data_vars['z']
xbnds = [ raster.x.data[500], raster.x.data[900] ]
ybnds = [ raster.y.data[600], raster.y.data[1000] ]
slice_args = { 'x': slice(*xbnds), 'y': slice(*ybnds) }

for iC in range(4):
    t0 = time.time()
    band: xa.DataArray = raster[iC].sel( **slice_args )
    test_val = band.data[200,200]
    print( f" Completed band read # {iC} in {time.time()-t0} sec, shape = {band.shape}, test_val = {test_val}" )


