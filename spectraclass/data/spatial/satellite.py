from spectraclass.data.spatial.tile.tile import Block
import cartopy.crs as ccrs
import geoviews.tile_sources as gts
import holoviews as hv
from holoviews.streams import SingleTap, DoubleTap
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type, Iterable
import os, traitlets as tl
import xarray as xa
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
        self.selection_dmap = hv.DynamicMap(self.select_points, streams=[self.tap_stream, self.double_tap_stream])

    def register_point_selection(self, x, y ):
        from spectraclass.model.labels import lm
        (x, y) = self.points_projection.transform_point(x, y, self.projection)
        lm().on_button_press(x, y)

    @exception_handled
    def select_points(self, x, y, x2, y2):
        from spectraclass.model.labels import lm

        if None not in [x, y]:
            self.register_point_selection( x, y )
        elif None not in [x2, y2]:
            self.register_point_selection( x2, y2 )
        else:
            return hv.Points([])

        points: List[Tuple[float,float,str]] = lm().getPoints( )
        points = self.projection.transform_points(x, y, self.points_projection )
        return hv.Points(points, vdims='class').opts( marker='+', size=12, line_width=3, angle=45, color='class', cmap=lm().labelmap )

    def panel( self ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block = tm().getBlock()
        self.points_projection = block.projection
        (xlim, ylim) = block.get_extent(self.projection)
        tile_source = gts.tile_sources.get("EsriImagery", None).opts(xlim=xlim, ylim=ylim, width=600, height=570)
        return tile_source * self.selection_dmap