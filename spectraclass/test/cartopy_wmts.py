import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
from cartopy.mpl.geoaxes import GeoAxes

#url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS'
#layer = 'World_Imagery'
url = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
layer = 'VIIRS_CityLights_2012'

tcrs = ccrs.epsg(3857)
gcrs = ccrs.PlateCarree()
# xbnds = (-8542199.547, -8538333.685 )
# ybnds = ( 4732692.184, 4736558.046 )
# extent = xbnds+ybnds

ax: GeoAxes = plt.axes(projection=tcrs)
wmts: SlippyImageArtist = ax.add_wmts(url, layer)
ax.set_extent((-15, 25, 35, 60),crs=gcrs)

#def print_size( *args ):
#   print( f"ASIZE: {wmts.get_size()}" )

# wmts.add_callback( print_size )

plt.title('WMTS test')
plt.show()
