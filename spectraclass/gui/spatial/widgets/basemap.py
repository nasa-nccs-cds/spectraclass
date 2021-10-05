import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

EsriImagery = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.jpg'

plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)


plt.show()

