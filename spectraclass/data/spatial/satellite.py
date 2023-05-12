from spectraclass.data.spatial.tile.tile import Block
import cartopy.crs as ccrs
import geoviews.tile_sources as gts
import holoviews as hv
import cartopy.crs as crs
from spectraclass.model.labels import LabelsManager, lm
from holoviews.streams import SingleTap, DoubleTap
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type, Iterable
import os, logging, numpy as np

import requests, traceback
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

def spm() -> "SatellitePlotManager":
    return SatellitePlotManager.instance()

class SatellitePlotManager(SCSingletonConfigurable):
    projection = ccrs.GOOGLE_MERCATOR

    def __init__(self):
        super(SatellitePlotManager, self).__init__()
        self.tap_stream = SingleTap( transient=True )
        self.double_tap_stream = DoubleTap( rename={'x': 'x2', 'y': 'y2'}, transient=True)
        self.selection_points = hv.DynamicMap(self.select_points, streams=[self.tap_stream, self.double_tap_stream])

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
    def select_points(self, x, y, x2, y2):
        if None not in [x, y]:
            lm().on_button_press( x, y )
        elif None not in [x2, y2]:
            lm().on_button_press( x, y )
        points: List[Tuple[float,float,str]] = lm().getPoints()
        [x, y] = [ np.array( [pt[idim] for pt in points] ) for idim in [0,1] ]
        points: np.ndarray = self.projection.transform_points(self.points_projection, x, y )
        lgm().log( f" @@SatellitePlotManager.select_points: {points.tolist()}")
        return hv.Points(points, vdims='class').opts( marker='+', size=12, line_width=3, angle=45, color='class', cmap=lm().labelmap )

    def register_point_selection(self, x, y ):
        (x, y) = self.points_projection.transform_point(x, y, self.projection)
        lm().on_button_press(x, y)

    def satextent(self, full_tile=False) -> Tuple[ Tuple[float,float], Tuple[float,float] ]:
        return self.tile_extent if full_tile else self.block_extent

    def selection_basemap(self, xlim: Tuple[float,float] = None, ylim: Tuple[float,float] = None, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        point_selection = kwargs.get( 'point_selection', False )
        if xlim is None: (xlim, ylim) = tm().getBlock().get_extent( self.projection )
        tile_source = gts.tile_sources.get("EsriImagery", None).opts(xlim=xlim, ylim=ylim, width=600, height=570)
        lgm().log(f" @@SatellitePlotManager.selection_basemap: xlim={xlim} ylim={ylim} point_selection={point_selection}")
        return tile_source * self.selection_points if point_selection else tile_source