from spectraclass.data.spatial.tile.tile import Block
from typing import List
from urllib import request
from io import BytesIO
from math import log, exp, tan, atan, ceil
from PIL import Image
import traitlets as tl
import requests, traceback
from spectraclass.model.base import SCConfigurable
import traitlets.config as tlc
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

def gpm() -> "GooglePlotManager":
    return GooglePlotManager.instance()

class GooglePlotManager(tlc.SingletonConfigurable, SCConfigurable):
    api_key = tl.Unicode("google/api_key").tag(config=True)

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1

    def __init__( self ):
        super(GooglePlotManager, self).__init__()
        self.figure = plt.figure(2)
        self.plot = None
        self.image = None
        self.block = None
        self.axes: Axes = self.figure.add_subplot(111)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.figure.set_constrained_layout_pads( w_pad=0., h_pad=0. )
        self.google_maps_zoom_level = 17
        self.google = None

    def setBlock(self, block: Block = None, type ='satellite'):
        from spectraclass.data.spatial.tile.manager import TileManager
        if block is None:
            tm = TileManager.instance()
            self.block: Block = tm.getBlock()
            print("  @@GPM:  Getting block from TileManager")
        else:
            if block == self.block:
                print( "  @@GPM:  Using existing block")
                return
            else:
                self.block = block
                print("  @@GPM:  Setting block")
        self.google = GoogleMaps( block, self.api_key )
        try:
            extent = block.extent(4326)
            print( f"Setting satellite image extent: {extent}, xlim = {block.xlim}, ylim = {block.ylim}")
            print(f"Google Earth block center coords: {(extent[2]+extent[3])/2},{(extent[1]+extent[0])/2}")
            self.image = self.google.get_tiled_google_map(type, extent, self.google_maps_zoom_level)
            self.plot: AxesImage = self.axes.imshow(self.image, extent=extent, alpha=1.0, aspect='auto' )
            self.axes.set_xlim(extent[0],extent[1])
            self.axes.set_ylim(extent[2],extent[3])
            self._mousepress = self.plot.figure.canvas.mpl_connect('button_press_event', self.onMouseClick )
            self.figure.canvas.draw_idle()
        except AttributeError:
            print( "Cant get spatial bounds for satellite image")
        except Exception:
            traceback.print_exc()

    def set_axis_limits( self, xlims, ylims ):
        if self.image is not None:
            xlims1, ylims1 = self.block.project_extent( xlims, ylims, 4326 )
            self.axes.set_xlim(*xlims1 )
            self.axes.set_ylim(*ylims1)
            print( f"Setting satellite image bounds: {xlims} {ylims} -> {xlims1} {ylims1}")
            self.figure.canvas.draw_idle()

    def onMouseClick(self, event):
        if event.xdata != None and event.ydata != None:
            if event.inaxes ==  self.axes:
                rightButton: bool = int(event.button) == self.RIGHT_BUTTON
                event = dict( event="pick", type="image", lat=event.ydata, lon=event.xdata, button=int(event.button), transient=rightButton )
                print( f"SatellitePlot Mouse-click: {event}")

    def mpl_update(self):
        self.figure.canvas.draw_idle()

    def gui(self):
        self.setBlock()
        return self.figure.canvas

class GoogleMaps():

    tau = 6.283185307179586
    DEGREE = tau / 360
    ZOOM_OFFSET = 8
    MAXSIZE = 640
    LOGO_CUTOFF = 32

    def __init__( self, block: Block, api_key: str ):
        self.api_key = api_key
        self.image_size = [ 800, 800 ]
        self.block = block

    def extent(self, epsg: int ):
        return self.block.extent( epsg )

    def get_google_map( self, type: str, zoom=14  ) -> Image.Image:
        extent = self.block.extent( 4326 )   # left, right, bottom, top
        center = [ (extent[0]+extent[1])/2, (extent[2]+extent[3])/2 ]
        url = f"http://maps.googleapis.com/maps/api/staticmap?center={center[0]},{center[1]}&size={self.image_size[0]}x{self.image_size[1]}&zoom={zoom}&sensor=false&key={self.api_key}&maptype={type}"
        print( f"Accessing google map at {center[0]},{center[1]} with dimensions {self.image_size[0]}x{self.image_size[1]}\n  ** url = {url}" )
        buffer = BytesIO(request.urlopen(url).read())
        google_image: Image.Image = Image.open(buffer)
        return google_image

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

    def get_tiled_google_map( self, type: str, extent: List[float], zoom=17 ) -> Image.Image:
        try:
            NW_lat_long = ( extent[3] * self.DEGREE, extent[0] * self.DEGREE )
            SE_lat_long = ( extent[2] * self.DEGREE, extent[1] * self.DEGREE )

            ullat, ullon = NW_lat_long
            lrlat, lrlon = SE_lat_long

            print( f" get_tiled_google_map: extent = lat:{[ullat,lrlat]}, lon:{[ullon,lrlon]}")

            # convert all these coordinates to pixels
            ulx, uly = self.latlon2pixels(ullat, ullon, zoom)
            lrx, lry = self.latlon2pixels(lrlat, lrlon, zoom)

            # calculate total pixel dimensions of final image
            dx, dy = lrx - ulx, uly - lry

            # calculate rows and columns
            cols, rows = ceil(dx / self.MAXSIZE), ceil(dy / self.MAXSIZE)

            # calculate pixel dimensions of each small image
            width = ceil(dx / cols)
            height = ceil(dy / rows)
            heightplus = height + self.LOGO_CUTOFF

            # assemble the image from stitched
            final: Image.Image = Image.new('RGB', (int(dx), int(dy)))
            for x in range(cols):
                for y in range(rows):
                    dxn = width * (0.5 + x)
                    dyn = height * (0.5 + y)
                    latn, lonn = self.pixels2latlon( ulx + dxn, uly - dyn - self.LOGO_CUTOFF / 2, zoom)
                    position = ','.join((str(latn / self.DEGREE), str(lonn / self.DEGREE)))
                    urlparams = {
                        'center': position,
                        'zoom': str(zoom),
                        'size': '%dx%d' % (width, heightplus),
                        'maptype': 'satellite',
                        'sensor': 'false',
                        'scale': 1
                    }
                    urlparams['key'] = self.api_key
                    urlparams['maptype'] = type
                    url = 'http://maps.google.com/maps/api/staticmap'
                    try:
                        response = requests.get(url, params=urlparams)
                        response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        print(e)
                        return None

                    im = Image.open(BytesIO(response.content))
                    final.paste(im, (int(x * width), int(y * height)))
        except Exception as err:
            print( f"get_tiled_google_map error: {err}")
            traceback.print_exc()
            return  Image.new( 'RGB', (0, 0) )

        return final