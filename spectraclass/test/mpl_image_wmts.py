import logging, os, traceback
from owslib.wmts import WebMapTileService
import cartopy.crs as ccrs
import os, conda, numbers, time, contextlib
from typing import List, Union, Tuple, Optional, Dict, Callable
import numpy as np
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.gui.spatial.image import TileServiceImage
from spectraclass.data.base import DataManager, dm
import xarray as xa
from multiprocessing import Process, freeze_support
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes
from spectraclass.gui.spatial.source import WMTSRasterSource
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import rioxarray as rio
from matplotlib.collections import QuadMesh
from spectraclass.xext.xgeo import XGeo

log_file = os.path.expanduser('~/.spectraclass/logging/geospatial.log')
file_handler = logging.FileHandler(filename=log_file, mode='w')
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    freeze_support()
    fig: plt.Figure = plt.figure( 0, figsize = (6, 6) )
    ax: Axes = fig.add_subplot(111)

    band_index = 100
    dmi: DataManager = DataManager.initialize( "demo2",'desis' )
    project_data: xa.Dataset = dmi.loadCurrentProject( "main" )
    block = tm().getBlock()
    print( block.extent() )
    # qmesh: QuadMesh = block.data[band_index].plot.imshow( ax, alpha=0.3 )

    tile_server_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS'
    tile_service = WebMapTileService( tile_server_url )
    layer = list(tile_service.contents.keys())[0]
    wmts = WMTSRasterSource(tile_service, layer)
    img = TileServiceImage( ax, wmts, TileManager.crs )

    plt.show()