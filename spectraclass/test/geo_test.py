#%%

import xarray as xa
import panel as pn
import geoviews as gv
from gui.spatial.widgets.scrap_heap.tiles import TileManager
import cartopy.crs as ccrs
from holoviews.core.spaces import DynamicMap
gv.extension('bokeh')
import holoviews as hv

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

slider = pn.widgets.FloatSlider( start=0.0, end=1.0, step=0.01, value=1.0, name="Alpha" )
desis_image: DynamicMap = data_array.hvplot.image( cmap='jet', tiles="EsriImagery", width=plot_size[0], height=plot_size[1], crs=crs )

fig = hv.render(desis_image)
slider.jslink('value', fig.renderers[-1].glyph, 'global_alpha' )
