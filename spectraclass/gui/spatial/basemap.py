import logging, os, traceback, contextlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import xarray as xa
from spectraclass.util.logs import LogManager, lgm
from spectraclass.data.spatial.tile.manager import TileManager
from cartopy.io.ogc_clients import WMTSRasterSource
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

    def __init__(self):
        super(TileServiceBasemap, self).__init__()
        self.tile_server_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS'
        self.crs: str = TileManager.crs
        self.tile_service: WebMapTileService = None
        self.layer: str = None
        self.basemap: AxesImage = None
        self.wmts: WMTSRasterSource = None


    # def set_extent(self, xr: List[float], yr: List[float], **kwargs):
    #     crs = kwargs.get( 'crs', self.crs )
    #     self.gax.set_extent( xr + yr, crs=crs )

    @exception_handled
    def setup_plot( self, xlim: Tuple[float,float], ylim: Tuple[float,float], **kwargs ):
        plt.ioff()
        fig_index = kwargs.pop('index',100)
        fig_size = kwargs.pop('size', (6, 6))
        title = kwargs.pop('title', 'Selection Region')
        use_basemap = kwargs.pop('basemap', True)
        self.figure: Figure = plt.figure( fig_index, figsize=fig_size )
        plt.ion()
        lgm().log( f"Projection = {self.crs}")

        self.figure.suptitle(title)
        self.gax: Axes = self.figure.add_axes( [0.01, 0.07, 0.98, 0.93], **kwargs )  # [left, bottom, width, height] GeoAxes: projection=self.crs,
        self.gax.xaxis.set_visible( False ); self.gax.yaxis.set_visible( False )
        if use_basemap:
            self.set_basemap( xlim, ylim ) # self.gax.add_wmts( self.tile_service, self.layer )
        else:
            self.gax.set_xbound( xlim[0], xlim[1] )
            self.gax.set_ybound( ylim[0], ylim[1] )
        self.sax: Axes = self.figure.add_axes([0.01, 0.01, 0.85, 0.05])  # [left, bottom, width, height]
        self.figure.canvas.toolbar_visible = True
        self.figure.canvas.header_visible = False

    @exception_handled
    def set_basemap(self, xlim: Tuple[float,float], ylim: Tuple[float,float] ):
        self.tile_service = WebMapTileService(self.tile_server_url)
        self.layer: str = list(self.tile_service.contents.keys())[0]
        self.wmts = WMTSRasterSource( self.tile_service, self.layer )
        self.basemap = TileServiceImage(self.gax, self.wmts, self.crs, xrange=xlim, yrange=ylim )

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


