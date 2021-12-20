import xarray as xa
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import logging, os
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.gui.spatial.basemap import TileServiceBasemap
from widgets.polygons import PolygonInteractor
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from spectraclass.data.base import DataManager, ModeDataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.model.labels import LabelsManager, lm
import rioxarray as rio
from matplotlib.collections import QuadMesh
from spectraclass.xext.xgeo import XGeo
from collections import OrderedDict
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.model.base import SCSingletonConfigurable
from functools import partial
from cartopy.mpl.geoaxes import GeoAxes
from spectraclass.widgets.polygons import PolygonInteractor
from spectraclass.gui.spatial.widgets.layers import LayersManager, Layer
from spectraclass.gui.spatial.basemap import TileServiceBasemap
import traitlets as tl
import os, time, ipywidgets as ipw
from ipympl.backend_nbagg import Canvas, Toolbar
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import types, pandas as pd
import xarray as xa
import numpy as np
from typing import List, Dict, Tuple, Optional
from spectraclass.data.spatial.manager import SpatialDataManager
import math, atexit, os, traceback
import pathlib
from spectraclass.widgets.slider import PageSlider
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.colors import Normalize
from matplotlib.backend_bases import PickEvent, MouseButton  # , NavigationToolbar2
from spectraclass.gui.control import UserFeedbackManager, ufm
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider


log_file = os.path.expanduser('~/.spectraclass/logging/geospatial.log')
file_handler = logging.FileHandler(filename=log_file, mode='w')
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

dm: DataManager = DataManager.initialize("demo2", 'desis')
dm.loadCurrentProject("main")
classes = [ ('Class-1', "cyan"), ('Class-2', "green"), ('Class-3', "magenta"), ('Class-4', "blue") ]
lm().setLabels(classes)

# qmesh: QuadMesh = tile.plot.imshow( ax=base.gax, alpha=0.3 ) # 'band', ax=

class MapManager(SCSingletonConfigurable):
    init_band = tl.Int(10).tag(config=True, sync=True)

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1

    def __init__( self, **kwargs ):   # class_labels: [ [label, RGBA] ... ]
        super(MapManager, self).__init__()
        self._debug = False
        self.currentFrame = 0
        self.block: Block = None
        self.use_model_data = False
        self.slider: Optional[PageSlider] = None
        self.image: Optional[AxesImage] = None

    @property
    def data(self) -> Optional[xa.DataArray]:
        from spectraclass.data.base import DataManager, dm
        if self.block is None: self.setBlock()
        block_data: xa.DataArray = self.block.data
        if self.use_model_data:
            reduced_data: xa.DataArray = dm().getModelData().transpose()
            dims = [reduced_data.dims[0], block_data.dims[1], block_data.dims[2]]
            coords = [(dims[0], reduced_data[dims[0]]), (dims[1], block_data[dims[1]]), (dims[2], block_data[dims[2]])]
            shape = [c[1].size for c in coords]
            raster_data = reduced_data.data.reshape(shape)
            return xa.DataArray(raster_data, coords, dims, reduced_data.name, reduced_data.attrs)
        else:
            return block_data

    @property
    def frame_data(self) -> np.ndarray:
        return self.data[ :, self.currentFrame].flatten().values()

    def setBlock( self, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager
        tm = TileManager.instance()
        self.block: Block = tm.getBlock()
        if self.block is not None:
            self.nFrames = self.data.shape[0]
            self.band_axis = kwargs.pop('band', 0)
            self.z_axis_name = self.data.dims[self.band_axis]
            self.x_axis = kwargs.pop('x', 2)
            self.x_axis_name = self.data.dims[self.x_axis]
            self.y_axis = kwargs.pop('y', 1)
            self.y_axis_name = self.data.dims[self.y_axis]

    def gui(self,**kwargs):
        self.setBlock()
        basemap = kwargs.pop('basemap', True)
        self.base = TileServiceBasemap()
        self.base.setup_plot( self.block.xlim, self.block.ylim, basemap=basemap, **kwargs )
        self.init_map()
        self.region_selection = PolygonInteractor( self.base.gax )
        return self.base.gax.figure

    def init_map(self):
        self.image_data: xa.DataArray = self.data[self.init_band, :, :].squeeze( drop=True )
        self.image: AxesImage = self.image_data.plot.imshow(ax=self.base.gax, alpha=0.3)

mm = MapManager()
app = mm.gui()
plt.show()

# def dist(x, y):
#     d = x - y
#     return np.sqrt(np.dot(d, d))
#
# class PolyRec:
#     epsilon = 5  # max pixel distance to count as a vertex hit
#
#     def __init__(self, polyId, ax,  x, y, c="grey", on_change: Callable = None ):
#         self.ax = ax
#         self.color = c
#         self.canvas = ax.figure.canvas
#         self.polyId = polyId
#         self.selected = False
#         xs, ys = np.array( [x,x] ), np.array( [y,y] )
#         self.poly = Polygon( np.column_stack([xs,ys]), animated=True, facecolor=self.color, closed=False )
#         x, y = zip(*self.poly.xy)
#         self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
#         if on_change: self.cid = self.poly.add_callback( on_change )
#         else: self.cid = None
#         ax.add_patch(self.poly)
#         ax.add_line(self.line)
#         self.indx = -1
#
#     def contains_point(self, event: MouseEvent ) -> bool:
#         return self.poly.contains_point( (event.x,event.y) )
#
#     def vertex_selected( self, event: MouseEvent ):
#         xy = np.asarray(self.poly.xy)
#         xyt = self.poly.get_transform().transform(xy)
#         xt, yt = xyt[:, 0], xyt[:, 1]
#         d = np.hypot(xt - event.x, yt - event.y)
#         indseq, = np.nonzero(d == d.min())
#         d0 = d[ indseq[0] ]
#         selected = (d0 < self.epsilon)
#         self.indx = indseq[0] if selected else -1
#         return ( self.indx > -1 )
#
#     def clear_vertex_selection(self):
#         self.indx = -1
#
#     def _update(self):
#         self.line.set_data(zip(*self.poly.xy))
#
#     def insert_point(self, event ):
#         x, y = event.xdata, event.ydata
#         self.poly.xy = np.row_stack( [ self.poly.xy, np.array( [x, y] ) ] )
#         self.draw()
#
#     def complete( self ):
#         self.poly.xy[-1] = self.poly.xy[0]
#         self.line.set_visible(False)
#         self.poly.set_closed(True)
#         self.ax.draw_artist(self.line)
#
#     def update(self):
#         self.line.set_data(zip(*self.poly.xy))
#         self.ax.draw_artist(self.poly)
#         self.ax.draw_artist(self.line)
#
#     def draw(self):
#         self.canvas.draw_idle()
#
#     def set_selected(self, selected: bool ):
#         self.selected = selected
#         self.line.set_visible(selected)
#         self.ax.draw_artist(self.line)
#
#     def drag_vertex(self, event ):
#         x, y = event.xdata, event.ydata
#         self.poly.xy[ self.indx ] = x, y
#         indx1 = self.poly.xy.shape[0]-1
#         if self.indx == 0:     self.poly.xy[indx1] = self.poly.xy[0]
#         if self.indx == indx1: self.poly.xy[0]     = self.poly.xy[indx1]
#
# class PolygonInteractor:
#
#     def __init__(self, ax):
#         self.ax = ax
#         self.polys: List[PolyRec] = []
#         self.prec: PolyRec = None
#         self.enabled = False
#         self.editing = False
#         self.creating = False
#         self.fill_color = "grey"
#         self.background = None
#         self.canvas = ax.figure.canvas
#         self.canvas.mpl_connect( 'key_press_event', self.on_key_press )
#         self.canvas.mpl_connect( 'draw_event', self.on_draw )
#         self.cids = []
#
#     def update_callbacks(self):
#         self.update_navigation()
#         if self.enabled:
#             self.cids.append( self.canvas.mpl_connect('button_press_event', self.on_button_press) )
#             self.cids.append( self.canvas.mpl_connect('button_release_event', self.on_button_release) )
#             self.cids.append( self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move) )
#         else:
#             for cid in self.cids:  self.canvas.mpl_disconnect(cid)
#             self.cids = []
#
#     def update_navigation(self):
#         tbar = self.canvas.toolbar
#         for cid in [ tbar._id_press, tbar._id_release, tbar._id_drag ]:
#             self.canvas.mpl_disconnect(cid)
#         if not self.enabled:
#             tbar._id_press   = self.canvas.mpl_connect( 'button_press_event',   tbar._zoom_pan_handler )
#             tbar._id_release = self.canvas.mpl_connect( 'button_release_event', tbar._zoom_pan_handler )
#             tbar._id_drag    = self.canvas.mpl_connect( 'motion_notify_event',   tbar.mouse_move )
#
#     @exception_handled
#     def set_enabled(self, enabled ):
#         if enabled != self.enabled:
#             self.enabled = enabled
#             lgm().log(f"set_enabled: {self.enabled}")
#             self.update_callbacks()
#
#     def add_poly( self, event ):
#         if not self.in_poly(event):
#             x, y = event.xdata, event.ydata
#             polyId = len(self.polys)
#             self.prec = PolyRec( polyId, self.ax, x, y, self.fill_color, self.poly_changed )
#             self.polys.append( self.prec )
#             self.creating = True
#         return self.prec
#
#     @exception_handled
#     def on_draw(self, event):
#         self.background = self.canvas.copy_from_bbox(self.ax.bbox)
#         for prec in self.polys:
#             self.ax.draw_artist(prec.poly)
#             self.ax.draw_artist(prec.line)
#
#     def poly_changed(self, poly):
#         if self.prec is not None:
#             vis = self.prec.line.get_visible()
#             Artist.update_from(self.prec.line, poly)
#             self.prec.line.set_visible(vis)
#
#     def in_poly( self, event ) -> Optional[PolyRec]:
#         for prec in self.polys:
#             if prec.contains_point( event ):
#                 return prec
#         return None
#
#     def select_poly(self, event):
#         self.prec = self.in_poly( event )
#         selected_pid = self.prec.polyId if (self.prec is not None) else -1
#         for prec in self.polys:
#             prec.set_selected( prec.polyId == selected_pid )
#         self.draw()
#
#     def close_poly(self):
#         self.prec.complete()
#         self.prec = None
#         self.creating = False
#         self.draw()
#
#     @exception_handled
#     def on_button_press(self, event: MouseEvent ):
#         if event.inaxes is None: return
#         print( event )
#
#         if event.button == 1:
#             if self.enabled:
#                 if event.dblclick:
#                     if self.creating:   self.close_poly()
#                     else:               self.select_poly( event )
#                 else:
#                     if self.prec is None:  self.add_poly( event )
#                     else:
#                         if self.creating:  self.prec.insert_point( event )
#                         else:              self.editing = self.prec.vertex_selected( event )
#
#         elif event.button == 3:
#             pass
#
#     @exception_handled
#     def on_button_release(self, event):
#         if self.prec is not None:
#             self.prec.clear_vertex_selection()
#             self.editing = False
#
#     @exception_handled
#     def on_key_press(self, event):
#         if not event.inaxes:
#             return
#         if   event.key == 'w':  self.set_enabled( True )
#         elif event.key == 'x':  self.set_enabled( False )
#
#     @exception_handled
#     def on_mouse_move(self, event):
#         if event.inaxes is None: return
#         if (self.editing or self.creating):
#             self.prec.drag_vertex( event )
#             self.draw()
#
#     @exception_handled
#     def draw(self):
#         if self.background is not None:
#             self.canvas.restore_region(self.background)
#         for prec in self.polys: prec.update()
#         self.canvas.blit(self.ax.bbox)



