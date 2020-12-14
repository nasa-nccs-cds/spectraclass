import numpy as np
import xarray as xa
import rioxarray as rio
from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage
from spectraclass.model.base import SCConfigurable
from spectraclass.data.spatial.tile import Block
from matplotlib.axes import Axes
from typing import List, Union, Dict, Callable, Tuple, Any
import collections.abc, traceback
from spectraclass.data.google import GoogleMaps
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


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

class LabelingCanvas(SCConfigurable):

    def __init__(self,  **kwargs ):
        SCConfigurable.__init__(self)
        self.figure = Figure()

    def setBlock(self, block_coords: Tuple[int], **kwargs   ):
        return self.console.setBlock( block_coords, **kwargs  )

    @property
    def button_actions(self) -> Dict[str,Callable]:
        return self.console.button_actions

    @property
    def menu_actions(self) -> Dict:
        return self.console.menu_actions

    def mpl_update(self):
        self.console.update_canvas()
        self.update()
        self.repaint()

    def getNewImage(self):
        return self.console.getNewImage()

    def getBlock(self) -> Block:
        return self.console.block

    def extent(self):
        return self.console.block.extent()

class SatellitePlotManager(QObject, EventClient):

    def __init__(self):
        QObject.__init__(self)
        self._gui = None

    def gui( self  ):
        if self._gui is None:
            self._gui = SatellitePlotCanvas( self.process_mouse_event )
            self.activate_event_listening()
        return self._gui

    def process_mouse_event(self, event ):
        self.submitEvent( event, EventMode.Gui )

    def processEvent( self, event ):
        if event.get('event') == "gui":
            if self._gui is not None:
                if event.get('type') == "zoom":
                    xlim, ylim = event.get('xlim'), event.get('ylim')
                    self._gui.set_axis_limits( xlim, ylim )

class SatellitePlotCanvas(FigureCanvas):

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1

    def __init__( self, eventProcessor ):
        self.figure = Figure( constrained_layout=True )
        FigureCanvas.__init__(self, self.figure )
        self.plot = None
        self.image = None
        self.block = None
        self._eventProcessor = eventProcessor
        FigureCanvas.setSizePolicy(self, QSizePolicy.Ignored, QSizePolicy.Ignored)
        FigureCanvas.setContentsMargins( self, 0, 0, 0, 0 )
        FigureCanvas.updateGeometry(self)
        self.axes: Axes = self.figure.add_subplot(111)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.figure.set_constrained_layout_pads( w_pad=0., h_pad=0. )
        self.google_maps_zoom_level = 17
        self.google = None

    def setBlock(self, block: Block, type ='satellite'):
        print(" SatelliteCanvas.setBlock ")
        self.block = block
        self.google = GoogleMaps(block)
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
                self._eventProcessor( event )

    def mpl_update(self):
        self.figure.canvas.draw_idle()

class ReferenceImageCanvas( FigureCanvas, EventClient ):

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1

    def __init__(self, parent, image_spec: Dict[str,Any], **kwargs ):
        self.figure = Figure( constrained_layout=True )
        FigureCanvas.__init__(self, self.figure )
        self.spec = image_spec
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Ignored, QSizePolicy.Ignored)
        FigureCanvas.setContentsMargins( self, 0, 0, 0, 0 )
        FigureCanvas.updateGeometry(self)
        self.axes: Axes = self.figure.add_subplot(111)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.figure.set_constrained_layout_pads( w_pad=0., h_pad=0. )
        self.image: xa.DataArray = rio.open_rasterio( self.spec['path'] )
        self.xdim = self.image.dims[-1]
        self.ydim = self.image.dims[-2]
        self.classes = [ ('Unlabeled', [1.0, 1.0, 1.0, 0.5]) ] + self.format_labels( self.spec.get( 'classes', [] ) )
        if self.classes == None:    cmap = "jet"
        else:                       cmap = ListedColormap( [ item[1] for item in self.classes ] )
        self.plot: AxesImage = self.axes.imshow( self.image.squeeze().values, alpha=1.0, aspect='auto', cmap=cmap  )
        self._mousepress = self.plot.figure.canvas.mpl_connect('button_press_event', self.onMouseClick)

    @classmethod
    def format_labels( cls, classes: List[Tuple[str, Union[str, List[Union[float, int]]]]]) -> List[Tuple[str, List[float]]]:
        from hyperclass.gui.labels import format_color
        return [(label, format_color(color)) for (label, color) in classes]

    def onMouseClick(self, event):
        if event.xdata != None and event.ydata != None:
            if event.inaxes ==  self.axes:
                coords = { self.xdim: event.xdata, self.ydim: event.ydata  }
                point_data = self.image.sel( **coords, method='nearest' ).values.tolist()
                ic = point_data[0] if isinstance( point_data, collections.abc.Sequence ) else point_data
                rightButton: bool = int(event.button) == self.RIGHT_BUTTON
                if rightButton: labelsManager.setClassIndex(ic)
                event = dict( event="pick", type="reference", y=event.ydata, x=event.xdata, button=int(event.button), transient=rightButton )
                if not rightButton: event['classification'] = ic
                self.submitEvent(event, EventMode.Gui)

    def mpl_update(self):
        self.figure.canvas.draw_idle()

    def computeClassificationError(self,  labels: xa.DataArray ):
        nerr = np.count_nonzero( self.image.values - labels.values )
        nlabels = np.count_nonzero( self.image.values > 0 )
        print( f"Classication errors: {nerr} errors out of {nlabels}, {(nerr*100.0)/nlabels:.2f}% error. ")

    def processEvent( self, event: Dict ):
        super().processEvent(event)
        if event.get('event') == 'gui':
            if event.get('type') == 'spread':
                labels: xa.Dataset = event.get('labels')
                self.computeClassificationError( labels )

satellitePlotManager = SatellitePlotManager()

