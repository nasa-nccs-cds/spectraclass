from spectraclass.data.spatial.tile.tile import Block
from typing import List
from urllib import request
from io import BytesIO
from math import log, exp, tan, atan, ceil
from PIL import Image
import os, traitlets as tl
import xarray as xa
import geoviews.tile_sources as gts
from geoviews.element.geo import WMTS
import requests, traceback
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled

def spm() -> "SatellitePlotManager":
    return SatellitePlotManager.instance()

class SatellitePlotManager(SCSingletonConfigurable):
    api_key = tl.Unicode("").tag(config=True)
    zoom_level = tl.Int(17).tag(config=True, sync=True)
    image_size = tl.Float(8.0).tag(config=True, sync=True)

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1

    def __init__( self ):
        super(SatellitePlotManager, self).__init__()
        self.image: Image.Image = None
        self.block: Block = None
        self.full_extent  = None
        self.map: EsriMaps = None

    @exception_handled
    def setBlock(self, block: Block = None, type ='satellite'):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        if block is None:
            self.block: Block = tm().getBlock()
            lgm().log("  @@GPM:  Getting block from TileManager")
        else:
            if block == self.block:
                lgm().log( "  @@GPM:  Using existing block")
                return
            else:
                self.block = block
                lgm().log("  @@GPM:  Setting block")
        self.map = EsriMaps( self.block )

    @property
    def cache_file_path(self):
        from spectraclass.data.base import DataManager, dm
        cache_file_dir = os.path.join( dm().cache_dir, dm().project_name, dm().mode, "google" )
        os.makedirs( cache_file_dir, 0o777, exist_ok = True )
        return os.path.join( cache_file_dir, f"{self.block.file_name}.tiff" )

    def set_axis_limits( self, xlims, ylims ):
        if self.image is not None:
            xlims1, ylims1 = self.block.project_extent( xlims, ylims, 4326 )
            lgm().log( f"Setting satellite image bounds: {xlims} {ylims} -> {xlims1} {ylims1}")

#     @exception_handled
#     def onMouseClick(self, event):
#         from spectraclass.application.controller import app
#         from spectraclass.data.spatial.tile.manager import TileManager, tm
#         lgm().log(f"SATELLITE.onMouseClick: {event}, in-axes = {event.inaxes==self.axes}")
#         if event.xdata != None and event.ydata != None:
#             if event.inaxes ==  self.axes:
# #                rightButton: bool = int(event.button) == self.RIGHT_BUTTON
#                 marker = tm().get_marker( event.xdata, event.ydata )
#                 app().add_marker( "satellite", marker )
# #                event = dict( event="pick", type="image", lat=event.ydata, lon=event.xdata, button=int(event.button), transient=rightButton )
# #                mm().processEvent( event )
#                 lgm().log( "Processed point selection.")

    def gui(self):
        self.setBlock()
        return self.map.get_map()
#
# class GoogleMaps():
#
#     tau = 6.283185307179586
#     DEGREE = tau / 360
#     ZOOM_OFFSET = 8
#     MAXSIZE = 640
#     LOGO_CUTOFF = 32
#
#     def __init__( self, block: Block, api_key: str ):
#         self.api_key = api_key
#         self.image_size = [ 800, 800 ]
#         self.block = block
#
#     def extent(self, epsg: int ):
#         return self.block.extent( epsg )
#
#     def get_google_map( self, type: str, zoom=14  ) -> Image.Image:
#         extent = self.block.extent( 4326 )   # left, right, bottom, top
#         center = [ (extent[0]+extent[1])/2, (extent[2]+extent[3])/2 ]
#         url = f"http://maps.googleapis.com/maps/api/staticmap?center={center[0]},{center[1]}&size={self.image_size[0]}x{self.image_size[1]}&zoom={zoom}&sensor=false&key={self.api_key}&maptype={type}"
#         lgm().log( f"Accessing google map at {center[0]},{center[1]} with dimensions {self.image_size[0]}x{self.image_size[1]}, key={self.api_key}\n  ** url = {url}" )
#         buffer = BytesIO(request.urlopen(url).read())
#         google_image: Image.Image = Image.open(buffer)
#         return google_image
#
#     @classmethod
#     def latlon2pixels(cls, lat, lon, zoom):
#         mx = lon
#         my = log(tan((lat + cls.tau / 4) / 2))
#         res = 2 ** (zoom + cls.ZOOM_OFFSET) / cls.tau
#         px = mx * res
#         py = my * res
#         return px, py
#
#     @classmethod
#     def pixels2latlon(cls, px, py, zoom):
#         res = 2 ** (zoom + cls.ZOOM_OFFSET) / cls.tau
#         mx = px / res
#         my = py / res
#         lon = mx
#         lat = 2 * atan(exp(my)) - cls.tau / 4
#         return lat, lon
#
#     @exception_handled
#     def get_tiled_google_map( self, type: str, extent: List[float], zoom=17 ) -> Image.Image:
#         NW_lat_long = ( extent[3] * self.DEGREE, extent[0] * self.DEGREE )
#         SE_lat_long = ( extent[2] * self.DEGREE, extent[1] * self.DEGREE )
#
#         ullat, ullon = NW_lat_long
#         lrlat, lrlon = SE_lat_long
#
#         # convert all these coordinates to pixels
#         ulx, uly = self.latlon2pixels(ullat, ullon, zoom)
#         lrx, lry = self.latlon2pixels(lrlat, lrlon, zoom)
#
#         # calculate total pixel dimensions of final image
#         dx, dy = lrx - ulx, uly - lry
#
#         # calculate rows and columns
#         cols, rows = ceil(dx / self.MAXSIZE), ceil(dy / self.MAXSIZE)
#
#         # calculate pixel dimensions of each small image
#         width = ceil(dx / cols)
#         height = ceil(dy / rows)
#         heightplus = height + self.LOGO_CUTOFF
#
#         # assemble the image from stitched
#         lgm().log( f" get_tiled_google_map[{zoom}]: extent = lat:{[ullat,lrlat]}, lon:{[ullon,lrlon]}, dims = {[int(dx), int(dy)]}")
#         final: Image.Image = Image.new('RGB', (int(dx), int(dy)))
#         for x in range(cols):
#             for y in range(rows):
#                 dxn = width * (0.5 + x)
#                 dyn = height * (0.5 + y)
#                 latn, lonn = self.pixels2latlon( ulx + dxn, uly - dyn - self.LOGO_CUTOFF / 2, zoom)
#                 position = ','.join((str(latn / self.DEGREE), str(lonn / self.DEGREE)))
#                 urlparams = {
#                     'center': position,
#                     'zoom': str(zoom),
#                     'size': '%dx%d' % (width, heightplus),
#                     'maptype': 'satellite',
#                     'sensor': 'false',
#                     'scale': 1
#                 }
#                 urlparams['key'] = self.api_key
#                 urlparams['maptype'] = type
#                 url = 'http://maps.google.com/maps/api/staticmap'
#                 response = requests.get(url, params=urlparams)
#                 response.raise_for_status()
#                 im = Image.open(BytesIO(response.content))
#                 final.paste(im, (int(x * width), int(y * height)))
#
#         return final


class EsriMaps():

    tau = 6.283185307179586
    DEGREE = tau / 360
    ZOOM_OFFSET = 8
    MAXSIZE = 640
    LOGO_CUTOFF = 32

    def __init__( self, block: Block ):
        self.image_size = [ 800, 800 ]
        self.block = block
        self.tilemap: WMTS = gts.EsriImagery.opts(width=self.image_size[0], height=self.image_size[1])

    def extent(self, epsg: int ):
        return self.block.extent

    def get_map( self ) -> WMTS:
        xlim, ylim = self.block.extent   # left, right, bottom, top
        plot = self.tilemap.opts(xlim=xlim, ylim=ylim) # , projection=ccrs.GOOGLE_MERCATOR)
        return plot

    @classmethod
    def latlon2pixels(cls, lat, lon, zoom):
        mx = lon
        my = log(tan((lat + cls.tau / 4) / 2))
        res = 2 ** (zoom + cls.ZOOM_OFFSET) / cls.tau
        px = mx * res
        py = my * res
        return px, py

    @classmethod
    def pixels2latlon(cls, px, py, zoom):
        res = 2 ** (zoom + cls.ZOOM_OFFSET) / cls.tau
        mx = px / res
        my = py / res
        lon = mx
        lat = 2 * atan(exp(my)) - cls.tau / 4
        return lat, lon

