import collections, io, math, time, warnings, weakref
from xml.etree import ElementTree
from multiprocessing import cpu_count, get_context, Pool
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
from functools import partial
from spectraclass.util.logs import lgm, exception_handled
from PIL import Image
import numpy as np
import shapely.geometry as sgeom

try:
    from owslib.wms import WebMapService
    from owslib.wfs import WebFeatureService
    import owslib.util
    import owslib.wmts

    _OWSLIB_AVAILABLE = True
except ImportError:
    WebMapService = None
    WebFeatureService = None
    _OWSLIB_AVAILABLE = False

import cartopy.crs as ccrs
from cartopy.io import LocatedImage, RasterSource
from cartopy.img_transform import warp_array

_OWSLIB_REQUIRED = 'OWSLib is required to use OGC web services.'

# Hardcode some known EPSG codes for now.
# The order given here determines the preferred SRS for WMS retrievals.
_CRS_TO_OGC_SRS = collections.OrderedDict(
    [(ccrs.PlateCarree(), 'EPSG:4326'),
     (ccrs.Mercator.GOOGLE, 'EPSG:900913'),
     (ccrs.OSGB(approx=True), 'EPSG:27700')
     ])

# Standard pixel size of 0.28 mm as defined by WMTS.
METERS_PER_PIXEL = 0.28e-3

_WGS84_METERS_PER_UNIT = 2 * math.pi * 6378137 / 360

METERS_PER_UNIT = {
    'urn:ogc:def:crs:EPSG::27700': 1,
    'urn:ogc:def:crs:EPSG::900913': 1,
    'urn:ogc:def:crs:OGC:1.3:CRS84': _WGS84_METERS_PER_UNIT,
    'urn:ogc:def:crs:EPSG::3031': 1,
    'urn:ogc:def:crs:EPSG::3413': 1,
    'urn:ogc:def:crs:EPSG::3857': 1,
    'urn:ogc:def:crs:EPSG:6.18.3:3857': 1
}

_URN_TO_CRS = collections.OrderedDict(
    [('urn:ogc:def:crs:OGC:1.3:CRS84', ccrs.PlateCarree()),
     ('urn:ogc:def:crs:EPSG::4326', ccrs.PlateCarree()),
     ('urn:ogc:def:crs:EPSG::900913', ccrs.GOOGLE_MERCATOR),
     ('urn:ogc:def:crs:EPSG::27700', ccrs.OSGB(approx=True)),
     ('urn:ogc:def:crs:EPSG::3031', ccrs.Stereographic(
         central_latitude=-90,
         true_scale_latitude=-71)),
     ('urn:ogc:def:crs:EPSG::3413', ccrs.Stereographic(
         central_longitude=-45,
         central_latitude=90,
         true_scale_latitude=70)),
     ('urn:ogc:def:crs:EPSG::3857', ccrs.GOOGLE_MERCATOR),
     ('urn:ogc:def:crs:EPSG:6.18.3:3857', ccrs.GOOGLE_MERCATOR)
     ])

# XML namespace definitions
_MAP_SERVER_NS = '{http://mapserver.gis.umn.edu/mapserver}'
_GML_NS = '{http://www.opengis.net/gml}'


def _warped_located_image(image, source_projection, source_extent,
                          output_projection, output_extent, target_resolution):
    """
    Reproject an Image from one source-projection and extent to another.

    Returns
    -------
    LocatedImage
        A reprojected LocatedImage, the extent of which is >= the requested
        'output_extent'.

    """
    if source_projection == output_projection:
        extent = output_extent
    else:
        # Convert Image to numpy array (flipping so that origin
        # is 'lower').
        # Convert to RGBA to keep the color palette in the regrid process
        # if any
        img, extent = warp_array(np.asanyarray(image.convert('RGBA'))[::-1],
                                 source_proj=source_projection,
                                 source_extent=source_extent,
                                 target_proj=output_projection,
                                 target_res=np.asarray(target_resolution,
                                                       dtype=int),
                                 target_extent=output_extent,
                                 mask_extrapolated=True)

        # Convert arrays with masked RGB(A) values to non-masked RGBA
        # arrays, setting the alpha channel to zero for masked values.
        # This avoids unsightly grey boundaries appearing when the
        # extent is limited (i.e. not global).
        if np.ma.is_masked(img):
            img[:, :, 3] = np.where(np.any(img.mask, axis=2), 0,
                                    img[:, :, 3])
            img = img.data

        # Convert warped image array back to an Image, undoing the
        # earlier flip.
        image = Image.fromarray(img[::-1])

    return LocatedImage(image, extent)


def _target_extents(extent, requested_projection, available_projection):
    """
    Translate the requested extent in the display projection into a list of
    extents in the projection available from the service (multiple if it
    crosses seams).

    The extents are represented as (min_x, max_x, min_y, max_y).

    """
    # Start with the requested area.
    min_x, max_x, min_y, max_y = extent
    target_box = sgeom.box(min_x, min_y, max_x, max_y)

    # If the requested area (i.e. target_box) is bigger (or nearly bigger) than
    # the entire output requested_projection domain, then we erode the request
    # area to avoid re-projection instabilities near the projection boundary.
    buffered_target_box = target_box.buffer(requested_projection.threshold,
                                            resolution=1)
    fudge_mode = buffered_target_box.contains(requested_projection.domain)
    if fudge_mode:
        target_box = requested_projection.domain.buffer(
            -requested_projection.threshold)

    # Translate the requested area into the server projection.
    polys = available_projection.project_geometry(target_box,
                                                  requested_projection)

    # Return the polygons' rectangular bounds as extent tuples.
    target_extents = []
    for poly in polys:
        min_x, min_y, max_x, max_y = poly.bounds
        if fudge_mode:
            # If we shrunk the request area before, then here we
            # need to re-inflate.
            radius = min(max_x - min_x, max_y - min_y) / 5.0
            radius = min(radius, available_projection.threshold * 15)
            poly = poly.buffer(radius, resolution=1)
            # Prevent the expanded request going beyond the
            # limits of the requested_projection.
            poly = available_projection.domain.intersection(poly)
            min_x, min_y, max_x, max_y = poly.bounds
        target_extents.append((min_x, max_x, min_y, max_y))

    return target_extents

class WMTSRasterSource(RasterSource):
    """
    A WMTS imagery retriever which can be added to a map.

    Uses tile caching for fast repeated map retrievals.

    Note
    ----
        Requires owslib and Pillow to work.

    """

    _shared_image_cache = weakref.WeakKeyDictionary()
    """
    A nested mapping from WMTS, layer name, tile matrix name, tile row
    and tile column to the resulting PIL image::

        {wmts: {(layer_name, tile_matrix_name): {(row, column): Image}}}

    This provides a significant boost when producing multiple maps of the
    same projection or with an interactive figure.

    """

    def __init__(self, wmts, layer_name, gettile_extra_kwargs=None):
        """
        Parameters
        ----------
        wmts
            The URL of the WMTS, or an owslib.wmts.WebMapTileService instance.
        layer_name
            The name of the layer to use.
        gettile_extra_kwargs: dict, optional
            Extra keywords (e.g. time) to pass through to the
            service's gettile method.

        """
        self.image_cache = {}
        if WebMapService is None:
            raise ImportError(_OWSLIB_REQUIRED)

        if not (hasattr(wmts, 'tilematrixsets') and
                hasattr(wmts, 'contents') and
                hasattr(wmts, 'gettile')):
            wmts = owslib.wmts.WebMapTileService(wmts)

        try:
            layer = wmts.contents[layer_name]
        except KeyError:
            raise ValueError(
                f'Invalid layer name {layer_name!r} for WMTS at {wmts.url!r}')

        #: The OWSLib WebMapTileService instance.
        self.wmts = wmts

        #: The layer to fetch.
        self.layer = layer

        #: Extra kwargs passed through to the service's gettile request.
        if gettile_extra_kwargs is None:
            gettile_extra_kwargs = {}
        self.gettile_extra_kwargs = gettile_extra_kwargs

        self._matrix_set_name_map = {}

    def _matrix_set_name(self, target_projection):
        key = id(target_projection)
        matrix_set_name = self._matrix_set_name_map.get(key)
        if matrix_set_name is None:
            if hasattr(self.layer, 'tilematrixsetlinks'):
                matrix_set_names = self.layer.tilematrixsetlinks.keys()
            else:
                matrix_set_names = self.layer.tilematrixsets

            def find_projection(match_projection):
                result = None
                for tile_matrix_set_name in matrix_set_names:
                    matrix_sets = self.wmts.tilematrixsets
                    tile_matrix_set = matrix_sets[tile_matrix_set_name]
                    crs_urn = tile_matrix_set.crs
                    tms_crs = _URN_TO_CRS.get(crs_urn)
                    if tms_crs == match_projection:
                        result = tile_matrix_set_name
                        break
                return result

            # First search for a matrix set in the target projection.
            matrix_set_name = find_projection(target_projection)
            if matrix_set_name is None:
                # Search instead for a set in _any_ projection we can use.
                for possible_projection in _URN_TO_CRS.values():
                    # Look for supported projections (in a preferred order).
                    matrix_set_name = find_projection(possible_projection)
                    if matrix_set_name is not None:
                        break
                if matrix_set_name is None:
                    # Fail completely.
                    available_urns = sorted({
                        self.wmts.tilematrixsets[name].crs
                        for name in matrix_set_names})
                    msg = 'Unable to find tile matrix for projection.'
                    msg += f'\n    Projection: {target_projection}'
                    msg += '\n    Available tile CRS URNs:'
                    msg += '\n        ' + '\n        '.join(available_urns)
                    raise ValueError(msg)
            self._matrix_set_name_map[key] = matrix_set_name
        return matrix_set_name

    def validate_projection(self, projection):
        self._matrix_set_name(projection)

    @exception_handled
    def fetch_raster(self, projection, extent, target_resolution):
        matrix_set_name = self._matrix_set_name(projection)
        wmts_extents = [extent]
        width, height = target_resolution
        located_images = []
        for wmts_desired_extent in wmts_extents:
            # Calculate target resolution for the actual polygon.  Note that this gives *every* polygon enough pixels for the whole result, which is potentially excessive!
            min_x, max_x, min_y, max_y = wmts_desired_extent
            max_pixel_span = min((max_x - min_x) / width, (max_y - min_y) / height)
            t0 = time.time()
            wmts_image, wmts_actual_extent = self._wmts_images( self.wmts, self.layer, matrix_set_name, extent=wmts_desired_extent, max_pixel_span=max_pixel_span)
            t1 = time.time()
            located_image = LocatedImage(wmts_image, wmts_actual_extent)
            lgm().log( f" fetch_raster-> dt={t1-t0:.2f}: {wmts_desired_extent} -> {wmts_actual_extent} ")
            located_images.append(located_image)
        return located_images

    def _choose_matrix(self, tile_matrices, meters_per_unit, max_pixel_span):
        # Get the tile matrices in order of increasing resolution.
        tile_matrices = sorted(tile_matrices, key=lambda tm: tm.scaledenominator, reverse=True)
        # Find which tile matrix has the appropriate resolution.
        max_scale = max_pixel_span * meters_per_unit / METERS_PER_PIXEL
        for tm in tile_matrices:
            if tm.scaledenominator <= max_scale:
                return tm
        return tile_matrices[-1]

    def _tile_span(self, tile_matrix, meters_per_unit):
        pixel_span = (tile_matrix.scaledenominator *
                      (METERS_PER_PIXEL / meters_per_unit))
        tile_span_x = tile_matrix.tilewidth * pixel_span
        tile_span_y = tile_matrix.tileheight * pixel_span
        return tile_span_x, tile_span_y

    def _select_tiles(self, tile_matrix, tile_matrix_limits,
                      tile_span_x, tile_span_y, extent):
        # Convert the requested extent from CRS coordinates to tile
        # indices. See annex H of the WMTS v1.0.0 spec.
        # NB. The epsilons get rid of any tiles which only just
        # (i.e. one part in a million) intrude into the requested
        # extent. Since these wouldn't be visible anyway there's nothing
        # to be gained by spending the time downloading them.
        min_x, max_x, min_y, max_y = extent
        matrix_min_x, matrix_max_y = tile_matrix.topleftcorner
        epsilon = 1e-6
        min_col = int((min_x - matrix_min_x) / tile_span_x + epsilon)
        max_col = int((max_x - matrix_min_x) / tile_span_x - epsilon)
        min_row = int((matrix_max_y - max_y) / tile_span_y + epsilon)
        max_row = int((matrix_max_y - min_y) / tile_span_y - epsilon)
        # Clamp to the limits of the tile matrix.
        min_col = max(min_col, 0)
        max_col = min(max_col, tile_matrix.matrixwidth - 1)
        min_row = max(min_row, 0)
        max_row = min(max_row, tile_matrix.matrixheight - 1)
        # Clamp to any layer-specific limits on the tile matrix.
        if tile_matrix_limits:
            min_col = max(min_col, tile_matrix_limits.mintilecol)
            max_col = min(max_col, tile_matrix_limits.maxtilecol)
            min_row = max(min_row, tile_matrix_limits.mintilerow)
            max_row = min(max_row, tile_matrix_limits.maxtilerow)
        return min_col, max_col, min_row, max_row

    def get_image(self, wmts, layer, matrix_set_name, tile_matrix_id, img_key ):
        img: Image = self.image_cache.get(img_key)
        if img is None:
            tile = wmts.gettile( layer=layer.id, tilematrixset=matrix_set_name, tilematrix=str(tile_matrix_id),
                                    row=str(img_key[0]), column=str(img_key[1]), **self.gettile_extra_kwargs )
            img = Image.open( io.BytesIO( tile.read() ) )
            self.image_cache[img_key] = img
        return (img_key, img)

    def _wmts_images(self, wmts, layer, matrix_set_name, extent, max_pixel_span):
        """
        Add images from the specified WMTS layer and matrix set to cover
        the specified extent at an appropriate resolution.

        The zoom level (aka. tile matrix) is chosen to give the lowest
        possible resolution which still provides the requested quality.
        If insufficient resolution is available, the highest available
        resolution is used.

        Parameters
        ----------
        wmts
            The owslib.wmts.WebMapTileService providing the tiles.
        layer
            The owslib.wmts.ContentMetadata (aka. layer) to draw.
        matrix_set_name
            The name of the matrix set to use.
        extent
            Tuple of (left, right, bottom, top) in Axes coordinates.
        max_pixel_span
            Preferred maximum pixel width or height in Axes coordinates.

        """
        # Find which tile matrix has the appropriate resolution.
        tile_matrix_set = wmts.tilematrixsets[matrix_set_name]
        tile_matrices = tile_matrix_set.tilematrix.values()
        meters_per_unit = METERS_PER_UNIT[tile_matrix_set.crs]
        tile_matrix = self._choose_matrix(tile_matrices, meters_per_unit, max_pixel_span)
        # Determine which tiles are required to cover the requested extent.
        tile_span_x, tile_span_y = self._tile_span(tile_matrix, meters_per_unit)
        tile_matrix_set_links = getattr(layer, 'tilematrixsetlinks', None)
        if tile_matrix_set_links is None:
            tile_matrix_limits = None
        else:
            tile_matrix_set_link = tile_matrix_set_links[matrix_set_name]
            tile_matrix_limits = tile_matrix_set_link.tilematrixlimits.get( tile_matrix.identifier)
        min_col, max_col, min_row, max_row = self._select_tiles( tile_matrix, tile_matrix_limits, tile_span_x, tile_span_y, extent)

        # Find the relevant section of the image cache.
        tile_matrix_id = tile_matrix.identifier
        cache_by_wmts = WMTSRasterSource._shared_image_cache
        cache_by_layer_matrix = cache_by_wmts.setdefault(wmts, {})
        self.image_cache = cache_by_layer_matrix.setdefault((layer.id, tile_matrix_id), {})
        big_img = None
        n_rows = 1 + max_row - min_row
        n_cols = 1 + max_col - min_col
        nproc = 4 # cpu_count()
        lgm().log(f" ***** Fetch image extent {extent}, (n_rows,n_cols) = {[n_rows,n_cols]}, nproc={nproc} ")
        image_ids = [ (row, col) for row in range(min_row, max_row + 1) for col in range(min_col, max_col + 1) ]
        image_processor =  partial( self.get_image, wmts, layer, matrix_set_name, tile_matrix_id )
        with get_context("spawn").Pool( processes=nproc ) as p:
            image_tiles = p.map( image_processor, image_ids )

        for ((row,col),img) in image_tiles:
            if big_img is None:
                size = (img.size[0] * n_cols, img.size[1] * n_rows)
                big_img = Image.new('RGBA', size, (255, 255, 255, 255))
            top = (row - min_row) * tile_matrix.tileheight
            left = (col - min_col) * tile_matrix.tilewidth
            big_img.paste(img, (left, top))

        if big_img is None:
            img_extent = None
        else:
            matrix_min_x, matrix_max_y = tile_matrix.topleftcorner
            min_img_x = matrix_min_x + tile_span_x * min_col
            max_img_y = matrix_max_y - tile_span_y * min_row
            img_extent = (min_img_x, min_img_x + n_cols * tile_span_x, max_img_y - n_rows * tile_span_y, max_img_y)
        return big_img, img_extent

