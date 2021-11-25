import logging, os, traceback, contextlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import xarray as xa
from cartopy.io.ogc_clients import WMTSRasterSource
from spectraclass.util.logs import LogManager, lgm
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.gui.spatial.image import TileServiceImage
from typing import List, Optional, Dict, Tuple
from spectraclass.util.logs import LogManager, lgm, exception_handled
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from owslib.wmts import WebMapTileService
import cartopy.crs as ccrs
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.data.spatial.tile.manager import TileManager, tm

class TileServiceBasemap(SCSingletonConfigurable):

    def __init__(self, basemapid: str ):
        super(TileServiceBasemap, self).__init__()
        self.basemapid= basemapid
        self.tile_server_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS'
        self.crs: str = TileManager.crs
        self.tile_service: WebMapTileService = None
        self.layer: str = None
        self.basemap: AxesImage = None

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
        self.gax: Axes = self.figure.add_axes( [0.01, 0.07, 0.98, 0.93], **kwargs )  # [left, bottom, width, height] GeoAxes: projection=self.crs,
        print( f"gax class: {self.gax.__class__}" )
        self.gax.xaxis.set_visible( False ); self.gax.yaxis.set_visible( False )
        if self.basemapid is not None:
            print(f"Adding WMTS basemap")
            self.tile_service = WebMapTileService(self.tile_server_url)
            self.layer: str = list(self.tile_service.contents.keys())[0]
            self.basemap: AxesImage = self.add_wmts() # self.gax.add_wmts( self.tile_service, self.layer )
        self.sax: Axes = self.figure.add_axes([0.01, 0.01, 0.85, 0.05])  # [left, bottom, width, height]
        self.figure.canvas.toolbar_visible = True
        self.figure.canvas.header_visible = False

    @exception_handled
    def add_wmts(self) -> AxesImage:
        wmts = WMTSRasterSource( self.tile_service, self.layer)
        img = TileServiceImage(self.gax, wmts)
        with self.hold_limits():
            self.gax.add_image(img)
        return img

    @contextlib.contextmanager
    def hold_limits(self, hold=True):
        data_lim = self.gax.dataLim.frozen().get_points()
        view_lim = self.gax.viewLim.frozen().get_points()
        other = (self.gax.ignore_existing_data_limits, self.gax._autoscaleXon, self.gax._autoscaleYon)
        try:
            yield
        finally:
            if hold:
                self.gax.dataLim.set_points(data_lim)
                self.gax.viewLim.set_points(view_lim)
                ( self.gax.ignore_existing_data_limits, self.gax._autoscaleXon, self.gax._autoscaleYon ) = other


