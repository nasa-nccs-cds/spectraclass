import matplotlib.pyplot as plt
from spectraclass.data.base import DataManager
import xarray as xa
from spectraclass.xext.xgeo import XGeo
from spectraclass.data.spatial.manager import SpatialDataManager
from spectraclass.gui.spatial.basemap import TileServiceBasemap

iband = 100
dm: DataManager = DataManager.initialize( "demo2", 'desis' )
tile_raster: xa.DataArray = DataManager.instance().modal.readSpectralData()
[x0,x1,y0,y1] = SpatialDataManager.extent( tile_raster )

base = TileServiceBasemap()
base.setup_plot( (x0 ,x1), (y0 ,y1), standalone=True )
raster_layer = tile_raster[iband].squeeze( drop=True )
raster_layer.plot.imshow( ax=base.gax, alpha=0.4 )
plt.show()