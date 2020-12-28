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
# class LabelingCanvas(SCConfigurable):
#
#     def __init__(self,  **kwargs ):
#         SCConfigurable.__init__(self)
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

# class GooglePlotManager(SCConfigurable):
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
#         self.classes = [ ('Unlabeled', [1.0, 1.0, 1.0, 0.5]) ] + self.format_labels( self.spec.get( 'classes', [] ) )
#         if self.classes == None:    cmap = "jet"
#         else:                       cmap = ListedColormap( [ item[1] for item in self.classes ] )
#         self.plot: AxesImage = self.axes.imshow( self.image.squeeze().values, alpha=1.0, aspect='auto', cmap=cmap  )
#         self._mousepress = self.plot.figure.canvas.mpl_connect('button_press_event', self.onMouseClick)
#
#     @classmethod
#     def format_labels( cls, classes: List[Tuple[str, Union[str, List[Union[float, int]]]]]) -> List[Tuple[str, List[float]]]:
#         from hyperclass.gui.labels import format_color
#         return [(label, format_color(color)) for (label, color) in classes]
#
#     def onMouseClick(self, event):
#         if event.xdata != None and event.ydata != None:
#             if event.inaxes ==  self.axes:
#                 coords = { self.xdim: event.xdata, self.ydim: event.ydata  }
#                 point_data = self.image.sel( **coords, method='nearest' ).values.tolist()
#                 ic = point_data[0] if isinstance( point_data, collections.abc.Sequence ) else point_data
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
# satellitePlotManager = GooglePlotManager()
#
