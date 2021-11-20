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

from holoviews.core.spaces import DynamicMap
gv.extension('bokeh')
import holoviews as hv
from holoviews import opts
from bokeh.layouts import column
from hvplot.plotting.core import hvPlot

iband = 0
plot_size = [ 500, 500 ]
origin = "upper"
input_type = "nc"

SpectralDataTile = "/Users/tpmaxwel/Development/Data/desis/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.tif"
SpectralDataset = "/Users/tpmaxwel/Development/Data/raster/tiles/DESIS-HSI-L1C-DT0468853252_003-20200628T153803-V0210-SPECTRAL_IMAGE.nc"

tilemap: WMTS = gts.EsriImagery.opts(width=400, height=400 )

dataset: xa.Dataset = xa.open_dataset( SpectralDataset ) if input_type == "nc" else xa.open_rasterio( SpectralDataTile )
raster: xa.DataArray = dataset.data_vars['z'] # TileManager.to_standard_form( data, origin )
band: xa.DataArray = raster[iband]
feature_plot: hv.Image = hv.Image( raster[0] ).opts( cmap="jet")

crs = raster.attrs['ccrs']
kdims = [ Dimension(d) for d in band.dims ]
hvp: hvPlot = band.hvplot

dset = hvDataset( band, band.dims, "desis" )

feature_plot =   hvp.image( cmap="jet", width=plot_size[0], height=plot_size[1]  )

print(".")
#feature_plot: gv.Image = gv.Image( band, crs=crs, kdims=kdims ).opts( cmap="jet", width=plot_size[0], height=plot_size[1] )  # nodata


# print( hv.help(gv.Image) )