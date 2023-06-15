from spectraclass.data.spatial.tile.tile import Block
import cartopy.crs as ccrs
import holoviews as hv
import geoviews as gv
from geoviews.element.geo import WMTS
import cartopy.crs as crs
import panel as pn
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.data.spatial.tile.manager import TileManager, tm
from holoviews.streams import SingleTap, DoubleTap
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type, Iterable
import os, logging, numpy as np
from holoviews.streams import Stream, param

import requests, traceback
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

def spm() -> "SatellitePlotManager":
    return SatellitePlotManager.instance()

Bounds = Stream.define('Bounds', xylim=param.Tuple(default=(0.0, 1.0, 0.0, 1.0), doc='Image Axis Bounds: x0, x1, y0, y1') )

class SatellitePlotManager(SCSingletonConfigurable):
    projection = ccrs.GOOGLE_MERCATOR

    def __init__(self):
        super(SatellitePlotManager, self).__init__()
        self.double_tap_stream = DoubleTap( transient=True)
        self.bounds_stream: Stream = Bounds()
        self.selection_points = hv.DynamicMap(self.select_points, streams=[self.double_tap_stream])
        self.tile_source = hv.DynamicMap( self.get_image_basemap, streams=[self.bounds_stream] )
        self.block_source: gv.element.geo.WMTS = None
        pn.bind( self.set_extent, block_selection=tm().block_selection.param.index )

    @property
    def block(self) -> Block:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return tm().getBlock()

    @property
    def points_projection(self) -> crs.Projection:
        return self.block.projection

    @property
    def tile_extent(self) -> Tuple[ Tuple[float,float], Tuple[float,float] ]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return tm().tile.get_extent(self.projection)

    @property
    def block_extent(self) -> Tuple[ Tuple[float,float], Tuple[float,float] ]:
        return self.block.get_extent(self.projection)

    @exception_handled
    def select_points(self, x, y ):
        lm().on_button_press( x, y )
        points: List[Tuple[float,float,str]] = lm().getPoints()
        [x, y] = [ np.array( [pt[idim] for pt in points] ) for idim in [0,1] ]
        points: np.ndarray = self.projection.transform_points(self.points_projection, x, y )
        lgm().log( f"SPM: select_points: {points.tolist()}")
        return hv.Points(points, vdims='class').opts( marker='+', size=12, line_width=3, angle=45, color='class', cmap=lm().labelmap )

    def register_point_selection(self, x, y ):
        (x, y) = self.points_projection.transform_point(x, y, self.projection)
        lm().on_button_press(x, y)

    def satextent(self, full_tile=False) -> Tuple[ Tuple[float,float], Tuple[float,float] ]:
        return self.tile_extent if full_tile else self.block_extent

    def block_basemap(self, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        point_selection = kwargs.get( 'point_selection', False )
        b: Block = tm().getBlock()
        (xlim, ylim) = b.get_extent( self.projection )
        self.tile_source = tm().getESRIImageryServer( xlim=xlim, ylim=ylim, width=600, height=570 )
        lgm().log( f"SPM: block{b.index[1]}: ImageryServer extent {xlim} {ylim}")
        return self.tile_source * self.selection_points if point_selection else self.tile_source

    def get_image_basemap(self, bounds: Tuple[float,float,float,float], **kwargs):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        xlim, ylim = bounds[:2], bounds[2:]
        point_selection = kwargs.get( 'point_selection', False )
        tile_source: gv.element.geo.WMTS = tm().getESRIImageryServer( xlim=xlim, ylim=ylim, width=600, height=570 )
        print( f"get_image_basemap: bounds={bounds}, server={id(tile_source)}")
        return tile_source * self.selection_points if point_selection else self.tile_source

    def set_extent(self, block_selection: int ):
        if block_selection > 0:
            from spectraclass.data.spatial.tile.manager import TileManager, tm
            block: Block = tm().getBlock( bindex= tm().bi2c(block_selection) )
            (xlim, ylim) = block.get_extent(self.projection)
            self.bounds_stream.event( xylim = xlim+ylim )
        #    self.tile_source.apply.opts( xlim=xlim, ylim=ylim )
        #    self.tile_source.select( x=xlim, y=ylim )
            lgm().log( f"TM: set_extent block_selection={block_selection}  xlim={xlim}, ylim={ylim} ")

    #     self.tile_source = hv.DynamicMap(self.get_tile_source,
    #                                      streams=dict(block_index=tm().block_selection.param.value))
    #
    # def get_tile_source(self, block_index) -> gv.element.geo.Tiles:
    #     from spectraclass.data.spatial.tile.manager import TileManager, tm
    #     block: Block = tm().getBlock(bindex=block_index)
    #     (xlim, ylim) = block.get_extent(self.projection)
    #     self.tile_source.apply.opts(extents=(xlim[0], ylim[0], xlim[1], ylim[1]))  # (left, bottom, right, top)
    #

