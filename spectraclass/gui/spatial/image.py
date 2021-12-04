
import xarray as xa
import numpy as np
from matplotlib.axes import Axes
import contextlib, time
from typing import List, Optional, Dict, Tuple
from matplotlib.image import AxesImage
from spectraclass.util.logs import LogManager, lgm, exception_handled
import matplotlib.artist

def toXA( vname: str, nparray: np.ndarray, format="np", transpose = False ):
    gs: List[int] = [*nparray.shape]
    if nparray.ndim == 2:
        dims = ['y', 'x']
        coords = { dims[i]: np.array(range(gs[i])) for i in (0, 1) }
    elif nparray.ndim == 3:
        if transpose:
            nparray = nparray.reshape([gs[0] * gs[1], gs[2]]).transpose().reshape([gs[2], gs[0], gs[1]])
        dims = ['band', 'y', 'x']
        coords = { dims[i]: np.array(range(nparray.shape[i])) for i in (0, 1, 2) }
    else:
        raise Exception(f"Can't convert numpy->xa array with {nparray.ndim} dims")
    return xa.DataArray( nparray, coords, dims, vname, dict(transform=[1, 0, 0, 0, 1, 0], fileformat=format))


class TileServiceImage(AxesImage):

    def __init__(self, ax: Axes, raster_source, projection, **kwargs):
        self.raster_source = raster_source
        xrange = kwargs.pop('xrange',None)
        yrange = kwargs.pop('yrange', None)
        kwargs.setdefault('in_layout', False)
        super().__init__(ax, **kwargs)
        self.projection = projection
        self.cache = []

        self.axes.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.axes.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.on_release()
        if xrange is not None: self.axes.set_xbound( xrange[0], xrange[1] )
        if yrange is not None: self.axes.set_ybound( yrange[0], yrange[1] )

#        with self.hold_limits():
        self.axes.add_image( self )

    def on_press(self, event=None):
        self.user_is_interacting = True

    def on_release(self, event=None):
        self.user_is_interacting = False
        self.stale = True

    def get_window_extent(self, renderer=None):
        return self.axes.get_window_extent(renderer=renderer)

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        lgm().log("TileServiceImage.DRAW START")
        t0 = time.time()
        if not self.get_visible():
            return

        window_extent = self.axes.get_window_extent()
        [x1, y1], [x2, y2] = self.axes.viewLim.get_points()
        if not self.user_is_interacting:
            t1 = time.time()
            lgm().log("TileServiceImage.FETCH START")
            located_images = self.raster_source.fetch_raster( self.projection, extent=[x1, x2, y1, y2], target_resolution=(window_extent.width, window_extent.height))
            self.cache = located_images
            lgm().log(f"TileServiceImage.FETCH END, time = {time.time()-t1}")

        for img, extent in self.cache:
            self.set_array(img)
            with self.hold_limits():
                self.set_extent(extent)
            super().draw(renderer, *args, **kwargs)
        lgm().log(f"TileServiceImage.DRAW END, time = {time.time()-t0}")

    def can_composite(self):
        return False

    @contextlib.contextmanager
    def hold_limits( self ):
        data_lim = self.axes.dataLim.frozen().get_points()
        view_lim = self.axes.viewLim.frozen().get_points()
        other = (self.axes.ignore_existing_data_limits, self.axes._autoscaleXon, self.axes._autoscaleYon)
        try:
            yield
        finally:
            self.axes.dataLim.set_points(data_lim)
            self.axes.viewLim.set_points(view_lim)
            (self.axes.ignore_existing_data_limits, self.axes._autoscaleXon, self.axes._autoscaleYon) = other

# class LabelingWidget(QWidget):
#     def __init__(self, parent, **kwargs):
#         QWidget.__init__(self, parent, **kwargs)
#         self.setLayout(QVBoxLayout())
#         self.canvas = LabelingCanvas(self, **kwargs)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#         self.layout().addWidget(self.toolbar)
#         self.layout().addWidget(self.canvas)
#
#     def initPlots( self ) -> Optional[AxesImage]:
#         return self.canvas.console.initPlots()
#
#     @property
#     def spectral_plot(self):
#         return self.canvas.console.spectral_plot
#
#     def setBlock(self, block_coords: Tuple[int], **kwargs    ):
#         return self.canvas.setBlock( block_coords, **kwargs  )
#
#     def getNewImage(self):
#         return self.canvas.getNewImage()
#
#     def getTile(self):
#         return self.canvas.console.getTile()
#
#     def getBlock(self) -> Block:
#         return self.canvas.getBlock()
#
#     def extent(self):
#         return self.canvas.extent()
#
#     @property
#     def button_actions(self) -> Dict[str, Callable]:
#         return self.canvas.button_actions
#
#     @property
#     def menu_actions(self) -> Dict:
#         return self.canvas.menu_actions
#
#     def mpl_update(self):
#         self.canvas.mpl_update()
#         self.update()
#         self.repaint()
#
# class LabelingCanvas(SCSingletonConfigurable):
#
#     def __init__(self,  **kwargs ):
#         SCSingletonConfigurable.__init__(self)
#         self.figure = Figure()
#
#     def setBlock(self, block_coords: Tuple[int], **kwargs   ):
#         return self.console.setBlock( block_coords, **kwargs  )
#
#     @property
#     def button_actions(self) -> Dict[str,Callable]:
#         return self.console.button_actions
#
#     @property
#     def menu_actions(self) -> Dict:
#         return self.console.menu_actions
#
#     def mpl_update(self):
#         self.console.update_canvas()
#         self.update()
#         self.repaint()
#
#     def getNewImage(self):
#         return self.console.getNewImage()
#
#     def getBlock(self) -> Block:
#         return self.console.block
#
#     def extent(self):
#         return self.console.block.extent()

# class SatellitePlotManager(SCSingletonConfigurable):
#
#     def __init__(self):
#         QObject.__init__(self)
#         self._gui = None
#
#     def gui( self  ):
#         if self._gui is None:
#             self._gui = SatellitePlotCanvas( self.process_mouse_event )
#             self.activate_event_listening()
#         return self._gui
#
#     def process_mouse_event(self, event ):
#         self.submitEvent( event, EventMode.Gui )
#
#     def processEvent( self, event ):
#         if event.get('event') == "gui":
#             if self._gui is not None:
#                 if event.get('type') == "zoom":
#                     xlim, ylim = event.get('xlim'), event.get('ylim')
#                     self._gui.set_axis_limits( xlim, ylim )



# class ReferenceImageCanvas( FigureCanvas, EventClient ):
#
#     RIGHT_BUTTON = 3
#     MIDDLE_BUTTON = 2
#     LEFT_BUTTON = 1
#
#     def __init__(self, parent, image_spec: Dict[str,Any], **kwargs ):
#         self.figure = Figure( constrained_layout=True )
#         FigureCanvas.__init__(self, self.figure )
#         self.spec = image_spec
#         self.setParent(parent)
#         FigureCanvas.setSizePolicy(self, QSizePolicy.Ignored, QSizePolicy.Ignored)
#         FigureCanvas.setContentsMargins( self, 0, 0, 0, 0 )
#         FigureCanvas.updateGeometry(self)
#         self.axes: Axes = self.figure.add_subplot(111)
#         self.axes.get_xaxis().set_visible(False)
#         self.axes.get_yaxis().set_visible(False)
#         self.figure.set_constrained_layout_pads( w_pad=0., h_pad=0. )
#         self.image: xa.DataArray = rio.open_rasterio( self.spec['path'] )
#         self.xdim = self.image.dims[-1]
#         self.ydim = self.image.dims[-2]
#         self._classes = [ ('Unlabeled', [1.0, 1.0, 1.0, 0.5]) ] + self.format_labels( self.spec.get( '_classes', [] ) )
#         if self._classes == None:    cmap = "jet"
#         else:                       cmap = ListedColormap( [ item[1] for item in self._classes ] )
#         self.plot: AxesImage = self.axes.imshow( self.image.squeeze().values, alpha=1.0, aspect='auto', cmap=cmap  )
#         self._mousepress = self.plot.figure.canvas.mpl_connect('button_press_event', self.onMouseClick)
#
#     @classmethod
#     def format_labels( cls, _classes: List[Tuple[str, Union[str, List[Union[float, int]]]]]) -> List[Tuple[str, List[float]]]:
#         from hyperclass.gui.labels import format_color
#         return [(label, format_color(color)) for (label, color) in _classes]
#
#     def onMouseClick(self, event):
#         if event.xdata != None and event.ydata != None:
#             if event.inaxes ==  self.axes:
#                 coords = { self.xdim: event.xdata, self.ydim: event.ydata  }
#                 spectra = self.image.sel( **coords, method='nearest' ).values.tolist()
#                 ic = spectra[0] if isinstance( spectra, collections.abc.Sequence ) else spectra
#                 rightButton: bool = int(event.button) == self.RIGHT_BUTTON
#                 if rightButton: labelsManager.setClassIndex(ic)
#                 event = dict( event="pick", type="reference", y=event.ydata, x=event.xdata, button=int(event.button), transient=rightButton )
#                 if not rightButton: event['classification'] = ic
#                 self.submitEvent(event, EventMode.Gui)
#
#     def mpl_update(self):
#         self.figure.canvas.draw_idle()
#
#     def computeClassificationError(self,  labels: xa.DataArray ):
#         nerr = np.count_nonzero( self.image.values - labels.values )
#         nlabels = np.count_nonzero( self.image.values > 0 )
#         print( f"Classication errors: {nerr} errors out of {nlabels}, {(nerr*100.0)/nlabels:.2f}% error. ")
#
#     def processEvent( self, event: Dict ):
#         super().processEvent(event)
#         if event.get('event') == 'gui':
#             if event.get('type') == 'spread':
#                 labels: xa.Dataset = event.get('labels')
#                 self.computeClassificationError( labels )
#
# satellitePlotManager = SatellitePlotManager()
#
