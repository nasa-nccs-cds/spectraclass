import logging, os, traceback
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import xarray as xa
from spectraclass.util.logs import LogManager, lgm
from spectraclass.data.spatial.tile.manager import TileManager
from typing import List, Optional, Dict, Tuple
from spectraclass.util.logs import LogManager, lgm, exception_handled
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from owslib.wmts import WebMapTileService
import cartopy.crs as ccrs
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.data.spatial.tile.manager import TileManager, tm

class TileServiceBasemap(SCSingletonConfigurable):

    def __init__(self):
        super(TileServiceBasemap, self).__init__()
        self.tile_server_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS'
        self.crs: str = TileManager.crs
        self.tile_service = WebMapTileService( self.tile_server_url )
        self.layer: str = list(self.tile_service.contents.keys())[0]

    # def set_extent(self, xr: List[float], yr: List[float], **kwargs):
    #     crs = kwargs.get( 'crs', self.crs )
    #     self.gax.set_extent( xr + yr, crs=crs )

    @exception_handled
    def setup_plot( self, **kwargs ):
        plt.ioff()
        fig_index = kwargs.pop('index',100)
        fig_size = kwargs.pop('size', (6, 6))
        title = kwargs.pop('title', 'Selection Region')
        self.figure: Figure = plt.figure( fig_index, figsize=fig_size )
        plt.ion()
        lgm().log( f"Projection = {self.crs}")

        self.figure.suptitle(title)
        self.gax:   GeoAxes = self.figure.add_axes( [0.01, 0.07, 0.98, 0.93], projection=self.crs, **kwargs )  # [left, bottom, width, height]
        self.gax.xaxis.set_visible( True ); self.gax.yaxis.set_visible( True )
        self.gax.add_wmts(self.tile_service, self.layer)
        self.sax: Axes = self.figure.add_axes([0.01, 0.01, 0.85, 0.05])  # [left, bottom, width, height]
        self.figure.canvas.toolbar_visible = True
        self.figure.canvas.header_visible = False

