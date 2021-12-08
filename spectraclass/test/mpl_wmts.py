import logging, os, traceback
from owslib.wmts import WebMapTileService
import cartopy.crs as ccrs
import os, conda, numbers, time, contextlib
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.gui.spatial.source import WMTSRasterSource
from spectraclass.gui.spatial.image import TileServiceImage
import numpy as np
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.base import DataManager, dm
import xarray as xa
from matplotlib.image import AxesImage
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes
from spectraclass.xext.xgeo import XGeo

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import rioxarray as rio

log_file = os.path.expanduser('~/.spectraclass/logging/geospatial.log')
file_handler = logging.FileHandler(filename=log_file, mode='w')
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

fig: plt.Figure = plt.figure( 0, figsize = (6, 6) )
ax: Axes = fig.add_subplot(111)
xbnds = [-8542199.547, -8538333.685 ]
ybnds = [ 4732692.184,  4736558.046 ]

tile_server_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS'
crs = ccrs.epsg(3857)
tile_service = WebMapTileService( tile_server_url )
layer = list(tile_service.contents.keys())[0]
print( layer )
wmts = WMTSRasterSource(tile_service, layer)
img = TileServiceImage( ax, wmts, crs, xrange=xbnds, yrange=ybnds )

plt.show()