from cartopy.io.shapereader import Reader as ShapeReader
from gui.spatial.widgets.scrap_heap.tiles import TileManager
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
from cartopy import crs as ccrs
import numpy as np

def get_extent( reader: ShapeReader, ssrs: ccrs.CRS, tsrs: ccrs.CRS ):
    np_extent = np.array( [ r.bounds for r in reader.records() ] )
    pt0 = tsrs.transform_point( np_extent[ :, 0 ].min(), np_extent[ :, 1 ].min(), ssrs )
    pt1 = tsrs.transform_point( np_extent[ :, 2 ].max(), np_extent[ :, 3 ].max(), ssrs )
    return [ pt0[0], pt1[0], pt0[1], pt1[1] ]

epsg = 4326
boundaries_shp = "/Users/tpmaxwel/Development/Data/gis/Maryland/Maryland_Physical_Boundaries_-_County_Boundaries_(Generalized).shp"
tiles_shp = "/Users/tpmaxwel/Development/Data/desis/desis_tile_srs.shp"
images_reader = ShapeReader( tiles_shp )
boundaries_geo = ShapeReader( boundaries_shp )
recs = boundaries_geo.records()
attrs = next(recs).attributes

boundaries_crs: ccrs.CRS = TileManager.get_shp_crs( boundaries_shp )
images_crs: ccrs.CRS = ccrs.epsg(4326)

crs: ccrs.CRS = ccrs.PlateCarree()
np_extent = get_extent( images_reader, images_crs, crs )
ax: GeoAxes = plt.axes( projection=crs )
ax.set_extent( np_extent )
ax.add_geometries( images_reader.geometries(), crs=images_crs, alpha=0.5, facecolor="blue" )
ax.add_geometries( boundaries_geo.geometries(), crs=boundaries_crs, alpha=0.6, facecolor="green" )
ax.gridlines( crs=ccrs.PlateCarree(), draw_labels=True )

# (x_limits=[166021.44308054057, 833978.5569194609], y_limits=[0.0, 9329005.182447437])

plt.show()



