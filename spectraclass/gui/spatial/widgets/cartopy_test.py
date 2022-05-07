from cartopy.io.img_tiles import GoogleTiles
import sys, os, datetime, platform
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
import numpy as np

fig = plt.figure(figsize=(5,5))

tiles = GoogleTiles( style="satellite", url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.jpg" )
ax = plt.axes(projection=tiles.crs)
ax.set_extent(( 153.05, 153.15, -26.55, -26.45)) # -67.875, -66.458, -19.220, -18.175
zoom = 14
ax.add_image( tiles, zoom )

plt.show()