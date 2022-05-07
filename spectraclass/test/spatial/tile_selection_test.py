import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

import xarray as xa
from matplotlib.backend_bases import MouseEvent
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from gui.spatial.widgets.scrap_heap.tiles import TileSelector, TileManager
import cartopy.crs as ccrs

iband = 0
block_size = 100
origin = "upper"
blocks_per_tile = 5
tile_size = block_size*blocks_per_tile
LabelDataFile = "/Users/tpmaxwel/GDrive/Tom/Data/ChesapeakeLandUse/CalvertCounty/CALV_24009_LandUse.tif"

tmgr = TileManager( LabelDataFile, tile_size, origin, nodata_fill=-1 )
crs: ccrs.CRS = tmgr.crs()

fig = plt.figure( figsize=(16,8) )
ax0: Axes = fig.add_subplot( 121, projection=crs )
tile_array0 = tmgr.get_tile( None, origin )
tile_data = tile_array0.data[iband]
vr = tmgr.vrange()
img0: AxesImage = ax0.imshow( tile_data, transform=crs, origin=origin, cmap="tab20", extent=tile_array0.attrs["extent"], vmin=vr[0], vmax=vr[1] )

def on_tile_selection( event: MouseEvent ):
    dloc = [ event.xdata, event.ydata ]
    tile_array = tmgr.get_tile( dloc, origin )
    tile_ext = tile_array.attrs["extent"]
    img0.set_extent( tile_ext )
    tile_data = tile_array.data[iband]
    img0.set_data( tile_data )
    img0.figure.canvas.draw()
    img0.figure.canvas.flush_events()

downscaled_data: xa.DataArray = tmgr.downscale( block_size, origin )
ax1 = fig.add_subplot( 122   )
img1: AxesImage = ax1.imshow( downscaled_data.data[iband], origin='upper', cmap="tab20", vmin=vr[0], vmax=vr[1] )
img1.set_extent( downscaled_data.extent )
rsize = tmgr.dsize( downscaled_data.transform, blocks_per_tile )
ts = TileSelector( ax1, rsize, on_tile_selection )
def onresize( event ): print( f" RESIZE: {event}" )
cid = fig.canvas.mpl_connect('resize_event', onresize)
ts.activate()
plt.show()
