import os, time
import matplotlib.pyplot as plt
from  cartopy import crs as ccrs
t0 = time.time()

fig = plt.figure(figsize=(8, 12))
cartopy_data_dir="/Users/tpmaxwel/opt/anaconda3/envs/jlab3/lib/python3.8/site-packages/cartopy/data"
fname = os.path.join( cartopy_data_dir, 'raster', 'sample', 'Miriam.A2012270.2050.2km.jpg' )
img_extent = (-120.67660000000001, -106.32104523100001, 13.2301484511245, 30.766899999999502)
img = plt.imread(fname)
gcrs = ccrs.PlateCarree()
mcrs = ccrs.Mollweide( -115.0 )

ax = plt.axes(projection=mcrs)
plt.title( 'Hurricane Miriam from the Aqua/MODIS satellite\n 2012 09/26/2012 20:50 UTC' )

# set a margin around the data
ax.set_xmargin(0.05)
ax.set_ymargin(0.10)

# add the image. Because this image was a tif, the "origin" of the image is in the
# upper left corner
ax.imshow(img, origin='upper', extent=img_extent, transform=gcrs)
ax.set_extent( img_extent, crs=gcrs )
# ax.coastlines(resolution='50m', color='black', linewidth=1)

print( f"Completed projection in {time.time()-t0} sec, shape = {0}")
plt.show()