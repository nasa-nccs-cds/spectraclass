from.tile import Tile, Block
from typing import List, Union, Dict, Callable, Tuple, Optional
from urllib import request
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from math import log, exp, tan, atan, ceil
from PIL import Image
import traitlets as tl
import traitlets.config as tlc
import sys, math, requests, traceback
from spectraclass.model.base import SCConfigurable

class GoogleMaps(SCConfigurable):
    api_key = tl.Unicode("google/api_key").tag(config=True)

    tau = 6.283185307179586
    DEGREE = tau / 360
    ZOOM_OFFSET = 8
    MAXSIZE = 640
    LOGO_CUTOFF = 32

    def __init__( self, block: Block ):
        self.image_size = [ 800, 800 ]
        self.figure: Figure = None
        self.ax = None
        self.block = block

    def extent(self, epsg: int ):
        return self.block.extent( epsg )

    def get_google_map( self, type: str, zoom=14  ):
        extent = self.block.extent( 4326 )   # left, right, bottom, top
        center = [ (extent[0]+extent[1])/2, (extent[2]+extent[3])/2 ]
        url = f"http://maps.googleapis.com/maps/api/staticmap?center={center[0]},{center[1]}&size={self.image_size[0]}x{self.image_size[1]}&zoom={zoom}&sensor=false&key={self.api_key}&maptype={type}"
        print( f"Accessing google map at {center[0]},{center[1]} with dimensions {self.image_size[0]}x{self.image_size[1]}\n  ** url = {url}" )
        buffer = BytesIO(request.urlopen(url).read())
        google_image: Image.Image = Image.open(buffer)
        if self.figure is not None: self.figure.remove()
        self.figure, self.ax = plt.subplots(1, 1)
        self.ax.imshow(google_image, extent=extent, alpha=1.0)
        plt.show( block = False )

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