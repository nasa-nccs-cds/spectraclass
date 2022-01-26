import logging, os, traceback, contextlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import xarray as xa
from spectraclass.util.logs import LogManager, lgm
from spectraclass.data.spatial.tile.manager import TileManager
from matplotlib.font_manager import FontProperties
from spectraclass.gui.spatial.source import WMTSRasterSource
from spectraclass.gui.spatial.image import TileServiceImage
from typing import List, Optional, Dict, Tuple
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from owslib.wmts import WebMapTileService
import cartopy.crs as ccrs
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.data.spatial.tile.manager import TileManager, tm

class TileServiceBasemap(SCSingletonConfigurable):

    def __init__(self, **kwargs ):
        super(TileServiceBasemap, self).__init__()
        self.tile_server_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/WMTS'
        self.crs: str = TileManager.crs
        self.tile_service: WebMapTileService = None
        self.layer: str = None
        self.basemap: TileServiceImage = None
        self.wmts: WMTSRasterSource = None
        self._block_selection = kwargs.get( 'block_selection', False )


    # def set_extent(self, xr: List[float], yr: List[float], **kwargs):
    #     crs = kwargs.get( 'crs', self.crs )
    #     self.gax.set_extent( xr + yr, crs=crs )

    @exception_handled
    def setup_plot( self, title: str, xlim: Tuple[float,float], ylim: Tuple[float,float], **kwargs ):
        standalone = kwargs.pop( 'standalone', False )
        if not standalone: plt.ioff()
        fig_index = kwargs.pop('index',100)
        fig_size = kwargs.pop('size', (6, 6))
        use_basemap = kwargs.pop('basemap', True)
        parallel = kwargs.pop('parallel', True)
        use_slider = kwargs.pop( 'slider', True )
        self.figure: Figure = plt.figure( fig_index, figsize=fig_size )
        self.figure.suptitle( title, color="yellow" )
        if not standalone: plt.ion()

        bounds = [0.01, 0.07, 0.98, 0.93] if use_slider else [0.01, 0.01, 0.98, 0.98]
        self.gax: Axes = self.figure.add_axes( bounds, **kwargs )  # [left, bottom, width, height] # , projection=self.crs
        self.gax.xaxis.set_visible( False ); self.gax.yaxis.set_visible( False )
        self.gax.title.set_color("orange")
        if use_basemap:
            self.set_basemap( xlim, ylim, parallel=parallel, **kwargs)
        else:
            self.gax.set_xbound( xlim[0], xlim[1] )
            self.gax.set_ybound( ylim[0], ylim[1] )
        if use_slider:
            self.bsax: Axes  = self.figure.add_axes( [0.01, 0.01, 0.98, 0.05])  # [left, bottom, width, height]
            self.msax: Axes  = self.figure.add_axes( [0.01, 0.01, 0.98, 0.05]) # [left, bottom, width, height]
            self.msax.set_visible( False )
            self.bsax.set_visible( True  )
        self.figure.canvas.toolbar_visible = True
        self.figure.canvas.header_visible = False
        return standalone

    def update(self):
        self.figure.canvas.draw_idle()

    def set_extent(self, extent: List[float] ):
        self.set_bounds( [extent[0],extent[1]], [extent[2],extent[3]] )

    def set_bounds(self, xrange: List[float], yrange: List[float] ):
        self.gax.set_xbound(*xrange)
        self.gax.set_ybound(*yrange)
        if self._block_selection: self.basemap.update_blocks()

    def set_alpha(self, alpha ):
        self.basemap.set_alpha( alpha )

    def gui(self):
        return self.figure.canvas

    @exception_handled
    def set_basemap(self, xlim: Tuple[float,float], ylim: Tuple[float,float], **kwargs ):
        self.tile_service = WebMapTileService(self.tile_server_url)
        self.layer: str = list(self.tile_service.contents.keys())[0]
        self.wmts = WMTSRasterSource( self.tile_service, self.layer, kwargs.pop('parallel',True) )
        self.basemap = TileServiceImage(self.gax, self.wmts, self.crs, xrange=xlim, yrange=ylim, block_selection=self._block_selection, **kwargs )

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


