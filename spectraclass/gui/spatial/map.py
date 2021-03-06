from collections import OrderedDict
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.model.base import SCSingletonConfigurable, Marker
from functools import partial
import traitlets as tl
from spectraclass.data.spatial.tile.tile import Block
from .satellite import SatellitePlotManager
from spectraclass.util.logs import LogManager, lgm, exception_handled
import pandas as pd
import xarray as xa
import numpy as np
from typing import List, Dict, Tuple, Optional
from spectraclass.data.spatial.manager import SpatialDataManager
import math, atexit, os, traceback
import pathlib
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.backend_tools import ToolToggleBase
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

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

def get_color_bounds( color_values: List[float] ) -> List[float]:
    color_bounds = []
    for iC, cval in enumerate( color_values ):
        if iC == 0: color_bounds.append( cval - 0.5 )
        else: color_bounds.append( (cval + color_values[iC-1])/2.0 )
    color_bounds.append( color_values[-1] + 0.5 )
    return color_bounds

def dms() -> SpatialDataManager:
    from spectraclass.data.base import DataManager, dm
    return dm().modal

# class ToggleDataSourceMode(ToolToggleBase):
#     image = os.path.join( ROOT_DIR, "icons", "module.png" )
#     description = "Toggle between input bands and reduced model"
#
#     def enable(self, event=None):
#         mm().set_data_source_mode( True )
#
#     def disable(self, event=None):
#         mm().set_data_source_mode( False )

class ToggleMarkersPlot(ToolToggleBase):
    description = "Toggle visibility of markers"
    image = "M"

    def enable(self, event=None):
        mm().setMarkersVisible( True )

    def disable(self, event=None):
        mm().setMarkersVisible( False )

class PageSlider(Slider):

    def __init__(self, ax: Axes, numpages = 10, valinit=0, valfmt='%1d', **kwargs ):
        self.facecolor=kwargs.get('facecolor',"yellow")
        self.activecolor = kwargs.pop('activecolor',"blue" )
        self.stepcolor = kwargs.pop('stepcolor', "#ff6f6f" )
        self.on_animcolor = kwargs.pop('on-animcolor', "#006622")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.maxIndexedPages = 24
        self.numpages = numpages
        self.axes = ax

        super(PageSlider, self).__init__(ax, "", 0, numpages, valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        indexMod = math.ceil( self.numpages / self.maxIndexedPages )
        for i in range(numpages):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = Rectangle((float(i)/numpages, 0), 1./numpages, 1, transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            if i % indexMod == 0:
                ax.text(float(i)/numpages+0.5/numpages, 0.5, str(i+1), ha="center", va="center", transform=ax.transAxes, fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = Button(bax, label='$\u25C1$', color=self.stepcolor, hovercolor=self.activecolor)
        self.button_forward = Button(fax, label='$\u25B7$', color=self.stepcolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self. forward)

    def refesh(self):
        self.axes.figure.canvas.draw()

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >=self.valmax: return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event=None):
        current_i = int(self.val)
        i = current_i+1
        if i >= self.valmax: i = self.valmin
        self.set_val(i)
        self._colorize(i)

    def backward(self, event=None):
        current_i = int(self.val)
        i = current_i-1
        if i < self.valmin: i = self.valmax -1
        self.set_val(i)
        self._colorize(i)

def mm() -> "MapManager":
    is_initialized = MapManager.initialized()
    mgr = MapManager.instance()
    if not is_initialized: mgr.observe( mgr.on_overlay_alpha_change, names=["overlay_alpha"] )
    return mgr

class MapManager(SCSingletonConfigurable):
    init_band = tl.Int(10).tag(config=True, sync=True)
    overlay_alpha = tl.Float(0.5).tag(config=True, sync=True)

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1

    def __init__( self, **kwargs ):   # class_labels: [ [label, RGBA] ... ]
        super(MapManager, self).__init__()
        self._debug = False
        self._cidpress = None
        self.currentFrame = 0
        self.block: Block = None
        self.slider: Optional[PageSlider] = None
        self.image: Optional[AxesImage] = None
        self.image_template: Optional[xa.DataArray]  = None
        self.overlay_image: Optional[AxesImage] = None
        self.use_model_data: bool = False
        self.labels = None
        self.transients = []
        self.plot_axes: Optional[Axes] = None
        self.marker_plot: Optional[PathCollection] = None
        self.label_map: Optional[xa.DataArray] = None
        self.dataLims = {}
        self.key_mode = None
        self.currentClass = 0
        self.nFrames = None
        self._adding_marker = False
        self.figure: Figure = plt.figure(100, figsize = (6, 6))
        self.labels_image: Optional[AxesImage] = None
        self.flow_iterations = kwargs.get( 'flow_iterations', 1 )
        self.frame_marker: Optional[Line2D] = None
        self.control_axes = {}
        self.setup_plot(**kwargs)

#        google_actions = [[maptype, None, None, partial(self.run_task, self.download_google_map, "Accessing Landsat Image...", maptype, task_context='newfig')] for maptype in ['satellite', 'hybrid', 'terrain', 'roadmap']]
        self.menu_actions = OrderedDict( Layers = [ [ "Increase Labels Alpha", 'Ctrl+>', None, partial( self.update_image_alpha, "labels", True ) ],
                                                    [ "Decrease Labels Alpha", 'Ctrl+<', None, partial( self.update_image_alpha, "labels", False ) ],
                                                    [ "Increase Band Alpha",   'Alt+>',  None, partial( self.update_image_alpha, "bands", True ) ],
                                                    [ "Decrease Band Alpha",   'Alt+<',  None, partial( self.update_image_alpha, "bands", False ) ] ] )

        atexit.register(self.exit)
        self._update(0)

    def gui(self):
        self.setBlock()
        return self.figure.canvas

    def refresh(self):
        self.setBlock()
        self.update_canvas()

    @property
    def toolbar(self):   #     -> NavigationToolbar2:
        return self.figure.canvas.toolbar

    @property
    def zeros(self):
        return self.image_template.copy( data = np.zeros( self.image_template.shape, np.int ) )

    @property
    def transform(self):
        return self.block.transform

    def point_coords( self, point_index: int ) -> Dict:
        block_data, point_data = self.block.getPointData()
        selected_sample: np.ndarray = point_data[ point_index ].values
        return dict( y = selected_sample[1], x = selected_sample[0] )

    def mark_point( self, pid: int, transient: bool ):
        cid, color = lm().selectedColor( not transient )
        marker = Marker( [pid], cid, labeled=False )
        self.add_marker( marker )

    @exception_handled
    def plot_overlay_image( self, image_data: np.ndarray = None ):
        if image_data is not None:
            lgm().log( f" plot image overlay, shape = {image_data.shape}, vrange = {[ image_data.min(), image_data.max() ]}, dtype = {image_data.dtype}" )
            self.overlay_image.set_data( image_data )
        self.overlay_image.set_alpha( self.overlay_alpha )
        self.update_canvas()

    def on_overlay_alpha_change(self, *args ):
        from spectraclass.gui.spatial.satellite import SatellitePlotManager, spm
        self.overlay_image.set_alpha( self.overlay_alpha )
        lgm().log(f" image overlay set alpha = {self.overlay_alpha}" )
        self.update_canvas()
        spm().plot_overlay_image( alpha = self.overlay_alpha )

    def setBlock( self, **kwargs ) -> Block:
        from spectraclass.data.spatial.tile.manager import TileManager
        self.clearLabels()
        reset = kwargs.get( 'reset', False )
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

            image = self.initPlots(**kwargs)
            if image is not None:
                self.add_slider(**kwargs)
                self.initLabels()
                self.update_plot_axis_bounds()
                self.plot_markers_image()
                self.update_plots()
                SatellitePlotManager.instance().setBlock(self.block)

        return self.block

    def update_plot_axis_bounds( self ):
        if self.plot_axes is not None:
            self.plot_axes.set_xlim( self.block.xlim )
            self.plot_axes.set_ylim( self.block.ylim )
            SatellitePlotManager.instance().set_axis_limits(self.block.xlim, self.block.ylim)


    # def computeMixingSpace(self, *args, **kwargs):
    #     labels: xa.DataArray = self.getExtendedLabelPoints()
    #     umapManager.computeMixingSpace( self.block, labels, **kwargs )
    #     self.plot_markers_volume()

#     def build_model(self, *args, **kwargs):
#         if self.block is None:
#             print( "Workflow violation: Must load a block before building model" )
#         else:
#             umapManager: ReductionManager = ReductionManager.instance()
#             labels: xa.DataArray = self.getExtendedLabelPoints()
#             umapManager.umap_embedding( labels=labels, **kwargs )
#
# #            self.plot_markers_volume()
#
#     def learn_classification( self, **kwargs  ):
#         if self.block is None:
#             print( "Workflow violation: Must load a block and spread some labels  before learning classification" )
#         else:
#             full_labels: xa.DataArray = self.getExtendedLabelPoints()
#             print( f"Learning Classification, labels shape = {full_labels.shape}, nLabels = {np.count_nonzero( full_labels > 0 )}")
#             event = dict(event="classify", type="learn", data=self.block, labels=full_labels )
# #            self.submitEvent( event, EventMode.Gui )
#
#     def apply_classification( self, **kwargs ):
#         print(f"Applying Classification")
#         event = dict( event="classify", type="apply", data=self.block  )
# #        self.submitEvent(event, EventMode.Gui )

    def initLabels(self):
        nodata_value = -2
        template = self.block.data[0].squeeze( drop=True )
        self.labels: xa.DataArray = xa.full_like( template, -1, dtype=np.dtype(np.int32) ).where( template.notnull(), nodata_value )
        self.labels.attrs['_FillValue'] = nodata_value
        self.labels.name = f"{self.block.data.name}_labels"
        self.labels.attrs[ 'long_name' ] = [ "labels" ]

    def clearLabels( self):
        if self.block is not None:
             self.initLabels()
             self.plot_markers_image()
             if self.labels_image is not None:
                self.labels_image.set_alpha(0.0)

    # def updateLabelsFromMarkers(self):
    #     lm().clearTransient()
    #     for marker in lm().markers:
    #         for pid in marker.pids:
    #             coords = self.block.pindex2coords(pid)
    #             index = self.block.coords2indices( coords['y'], coords['x'] )
    #             try:
    #                 self.labels[ index['iy'], index['ix'] ] = marker.cid
    #             except:
    #                 print( f"Skipping out of bounds label at local row/col coords {index['iy']} {index['ix']}")

    # def getLabeledPointData( self, update = True ) -> xa.DataArray:
    #     from spectraclass.data.base import DataManager, dm
    #     if update: self.updateLabelsFromMarkers()
    #     sdm: SpatialDataManager = dm().modal
    #     labeledPointData = sdm.raster2points( self.labels )
    #     return labeledPointData
    #
    # def getExtendedLabelPoints( self ) -> xa.DataArray:
    #     if self.label_map is None: return self.getLabeledPointData( True )
    #     return SpatialDataManager.raster2points( self.label_map )

    @property
    def data(self) -> Optional[xa.DataArray]:
        from spectraclass.data.base import DataManager, dm
        if self.block is None: return None
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

    @property
    def toolbarMode(self) -> str:
        return self.toolbar.mode

    @classmethod
    def time_merge( cls, data_arrays: List[xa.DataArray], **kwargs ) -> xa.DataArray:
        time_axis = kwargs.get('time',None)
        frame_indices = range( len(data_arrays) )
        merge_coord = pd.Index( frame_indices, name=kwargs.get("dim","time") ) if time_axis is None else time_axis
        result: xa.DataArray =  xa.concat( data_arrays, dim=merge_coord )
        return result

    def setup_plot(self, **kwargs):
        self.figure.suptitle("Band Image")
        self.plot_axes:   Axes = self.figure.add_axes([0.01, 0.07, 0.98, 0.93])  # [left, bottom, width, height]
        self.plot_axes.xaxis.set_visible( False ); self.plot_axes.yaxis.set_visible( False )
        self.slider_axes: Axes = self.figure.add_axes([0.01, 0.01, 0.85, 0.05])  # [left, bottom, width, height]
        self.figure.canvas.toolbar_visible = True
        self.figure.canvas.header_visible = False
        self.figure.canvas.resizable = True
        lgm().log( f"Canvas class = {self.figure.canvas.__class__}" )
        lgm().log( f"Canvas.manager class = {self.figure.canvas.manager.__class__}")
        items = self.figure.canvas.trait_values().items()
        for k,v in items: lgm().log(f" ** {k}: {v}")

    def invert_yaxis(self):
        self.plot_axes.invert_yaxis()

    def get_xy_coords(self,  ) -> Tuple[ np.ndarray, np.ndarray ]:
        return self.get_coord(self.x_axis ), self.get_coord( self.y_axis )

    def get_anim_coord(self ) -> np.ndarray:
        return self.get_coord( 0 )

    def get_coord(self,   iCoord: int ) -> np.ndarray:
        return self.data.coords[  self.data.dims[iCoord] ].values

    @exception_handled
    def create_image(self, **kwargs ) -> AxesImage:
        self.image_template: xa.DataArray =  self.data[ self.init_band, :, : ]
        nValid = np.count_nonzero(~np.isnan(self.image_template))
        lgm().log( f"\n ********* Creating Map Image, nValid={nValid}, data shape = {self.data.shape}, image shape = {self.image_template.shape}, band = {self.init_band}, data range = [ {self.data.min().values}, {self.data.max().values} ]")
        assert nValid > 0, "No valid pixels in image"
        colorbar = kwargs.pop( 'colorbar', False )
        image: AxesImage =  dms().plotRaster( self.image_template, ax=self.plot_axes, colorbar=colorbar, alpha=0.5, **kwargs )
        overlays = kwargs.get( "overlays", {} )
        for color, overlay in overlays.items():
            overlay.plot( ax=self.plot_axes, color=color, linewidth=2 )
        return image

    @exception_handled
    def add_tools(self):
        if self._cidpress is None:
            #        lgm().log( f"TOOLBAR: {self.figure.canvas.manager.toolbar.__class__}" )
            #        self.figure.canvas.manager.toolmanager.add_tool("ToggleSource", ToggleDataSourceMode)
            #        self.figure.canvas.manager.toolbar.add_tool("ToggleSource", 'navigation', 1)
            self.figure.canvas.manager.toolmanager.add_tool("ToggleMarkersPlot", ToggleMarkersPlot)
            self.figure.canvas.manager.toolbar.add_tool("ToggleMarkersPlot", 'navigation', 1)
            self.plot_axes.callbacks.connect('ylim_changed', self.on_lims_change)
            self._cidpress = self.figure.canvas.mpl_connect('button_press_event', self.onMouseClick)
            self._cidrelease = self.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease)

    @exception_handled
    def create_overlay_image( self ) -> AxesImage:
        assert self.image is not None, "Must create base image before overlay"
        overlay_image: AxesImage =  dms().plotRaster( self.image_template, itype='overlay', colorbar=False, alpha=0.0, ax=self.plot_axes, zeros=True )
        return overlay_image

    def on_lims_change(self, ax ):
         if ax == self.plot_axes:
             (x0, x1) = ax.get_xlim()
             (y0, y1) = ax.get_ylim()
             print(f"ZOOM Event: Updated bounds: ({x0},{x1}), ({y0},{y1})")

    def frame_color_pointcloud( self, **kwargs ) -> Optional[xa.DataArray]:
        from spectraclass.application.controller import app
        if self.currentFrame >= self.data.shape[0]: return None
        frame_data: xa.DataArray = self.data[ self.currentFrame ]
        lgm().log( f" color_pointcloud: currentFrame = {self.currentFrame}, frame data shape = {frame_data.shape}")
        app().color_pointcloud( frame_data.values.flatten(), **kwargs )
        return frame_data

    @exception_handled
    def update_plots(self):
        if self.image is not None:
            from spectraclass.data.base import DataManager, dm
            frame_data: xa.DataArray = self.frame_color_pointcloud()
            if frame_data is not None:
                self.image.set_data( frame_data.values  )
                drange = dms().get_color_bounds( frame_data )
                self.image.set_norm( Normalize( **drange ) )
                self.image.set_extent( self.block.extent() )
                plot_name = os.path.basename( dm().dsid() )
                lgm().log( f" Update Map: data shape = {frame_data.shape}, range = {drange}, extent = {self.block.extent()}")
                self.plot_axes.title.set_text(f"{plot_name}: Band {self.currentFrame+1}" )
                self.plot_axes.title.set_fontsize( 8 )

                if self.overlay_image is not None:
                    from spectraclass.gui.spatial.satellite import SatellitePlotManager, spm
                    self.clear_overlay_image()
                    spm().clear_overlay_image()
                self.update_canvas()

    def clear_overlay_image(self):
        self.overlay_image.set_extent(self.block.extent())
        self.overlay_image.set_alpha(0.0)

    def onMouseRelease(self, event):
        if event.inaxes ==  self.plot_axes:
             if   self.toolbarMode == "zoom rect":   self.toolbar.zoom()
             elif self.toolbarMode == "pan/zoom":    self.toolbar.pan()

        #         for listener in self.navigation_listeners:
        #             listener.set_axis_limits( self.plot_axes.get_xlim(), self.plot_axes.get_ylim() )

    @exception_handled
    def onMouseClick(self, event):
        if event.xdata != None and event.ydata != None:
            lgm().log(f"\nMouseClick event = {event}")
            if not self.toolbarMode and (event.inaxes == self.plot_axes) and (self.key_mode == None):
                rightButton: bool = int(event.button) == self.RIGHT_BUTTON
                pid = self.block.coords2pindex( event.ydata, event.xdata )
                if pid >= 0:
                    cid = lm().current_cid
                    lgm().log( f"Adding marker for pid = {pid}, cid = {cid}")
                    ptindices = self.block.pindex2indices(pid)
                    classification = self.label_map.values[ ptindices['iy'], ptindices['ix'] ] if (self.label_map is not None) else -1
                    self.add_marker( Marker( [pid], cid, classification = classification ) )
                    self.dataLims = event.inaxes.dataLim
                else:
                    lgm().log(f"Can't add marker, pid = {pid}")

    def set_data_source_mode(self, use_model_data: bool):
        self.use_model_data = bool
        lgm().log( f"Update data source: use_model_data = {self.use_model_data}" )
        fmsg = "Updating data source: " + ( "Using (reduced) model data" if self.use_model_data else "Using (raw) band data"  )
        ufm().show( fmsg, "yellow" )
        self.update_plots()
        ufm().clear()

    def add_marker(self, marker: Marker ):
        from spectraclass.application.controller import app
        if not self._adding_marker:
            self._adding_marker = True
            if marker is None:
                lgm().log( "NULL Marker: point select is probably out of bounds.")
            else:
                app().add_marker( "map", marker  )
        self._adding_marker = False

    # def undo_marker_selection(self, **kwargs ):
    #     if len( self.marker_list ):
    #         self.marker_list.pop()
    #         self.update_marker_plots( **kwargs )

    # def spread_labels(self, *args, **kwargs):
    #     if self.block is None:
    #         Task.taskNotAvailable( "Workflow violation", "Must load a block and label some points first", **kwargs )
    #     else:
    #         print( "Submitting training set" )
    #         labels: xa.DataArray = self.getLabeledPointData()
    #         sample_labels: Optional[xa.DataArray] = self.block.flow.spread( labels, self.flow_iterations, **kwargs )
    #         if sample_labels is not None:
    #             self.plot_label_map( sample_labels )

#     def plot_label_map( self, sample_labels: xa.DataArray, **kwargs ):
#         self.label_map: xa.DataArray =  sample_labels.unstack(fill_value=-2).astype(np.int32)
#         print( f"plot_label_map, labels shape = {self.label_map.shape}")
#         extent = dms().extent( self.label_map )
#         label_plot = self.label_map.where( self.label_map >= 0, 0 )
#         class_alpha = kwargs.get( 'alpha', 0.9 )
#         if self.labels_image is None:
#             label_map_colors: List = [ [ ic, label, list(color[0:3]) + [0.0 if (ic == 0) else class_alpha] ] for ic, (label, color) in enumerate( zip( lm().labels, lm().colors ) ) ]
#             self.labels_image = dms().plotRaster( label_plot, colors=label_map_colors, ax=self.plot_axes, colorbar=False )
#         else:
#             self.labels_image.set_data( label_plot.values )
#             self.labels_image.set_alpha( class_alpha )
#
#         self.labels_image.set_extent( extent )
#         self.update_canvas()
# #        event = dict( event="gui", type="update" )
# #        self.submitEvent(event, EventMode.Gui)

    def show_labels(self):
        if self.labels_image is not None:
            self.labels_image.set_alpha(1.0)
            self.update_canvas()

    def toggle_labels(self):
        if self.labels_image is not None:
            new_alpha = 1.0 if (self.labels_image.get_alpha() == 0.0) else 0.0
            self.labels_image.set_alpha( new_alpha )
            self.update_canvas()

    def get_layer(self, layer_id: str ):
        if layer_id == "bands": return self.image
        if layer_id == "labels": return self.labels_image
        raise Exception( f"Unrecognized layer: {layer_id}")

    def update_image_alpha( self, layer: str, increase: bool, *args, **kwargs ):
        image = self.get_layer( layer )
        if image is not None:
            current = image.get_alpha()
            if increase:   new_alpha = min( 1.0, current + 0.1 )
            else:          new_alpha = max( 0.0, current - 0.1 )
            print( f"Update Image Alpha: {new_alpha}")
            image.set_alpha( new_alpha )
            self.figure.canvas.draw_idle()

    # def clear_unlabeled(self):
    #     if self.marker_list:
    #         self.marker_list = [ marker for marker in self.marker_list if marker['c'] > 0 ]

    def get_markers( self ) -> Tuple[ List[float], List[float], List[List[float]] ]:
        ycoords, xcoords, colors, markers = [], [], [], lm().markers
        lgm().log(f" ** get_markers, #markers = {len(markers)}")
        for marker in markers:
            for pid in marker.pids:
                coords = self.block.pindex2coords( pid )
                if (coords is not None) and self.block.inBounds( coords['y'], coords['x'] ):   #  and not ( labeled and (c==0) ):
                    ycoords.append( coords['y'] )
                    xcoords.append( coords['x'] )
                    colors.append( lm().colors[marker.cid] )
                else:
                    lgm().log(f" ** coords[{pid}] out of bounds: {[coords['y'], coords['x']]}, bounds = ( {self.block._ylim}, {self.block._xlim} )")
                    lgm().log(f" ** Point coords range: {[coords['y'], coords['x']]}")
                    lgm().log(f" ** Projection bounds: xlim = {self.block.xlim}, ylim = {self.block.ylim} " )
                    yc = self.block.point_coords['y']; xc = self.block.point_coords['x']
                    lgm().log(f" ** Coordinates bounds: xrange = {[xc.min(),xc.max()]}, yrange = {[yc.min(),yc.max()]} ")
        return ycoords, xcoords, colors

    # def get_class_markers( self, **kwargs ) -> Dict[ int, List[int] ]:
    #     class_markers = {}
    #     for marker in lm().getMarkers():
    #         pids = class_markers.setdefault( marker.cid, [] )
    #         pids.extend( marker.pids )
    #     return class_markers

    @exception_handled
    def plot_markers_image( self ):
        if self.marker_plot:
            ycoords, xcoords, colors = self.get_markers()
            lgm().log(f" ** plot markers image, nmarkers = {len(ycoords)}")
            if len(ycoords) > 0:
                self.marker_plot.set_offsets(np.c_[xcoords, ycoords])
                self.marker_plot.set_facecolor(colors)
            else:
                offsets = np.ma.column_stack([[], []])
                self.marker_plot.set_offsets(offsets)
            self.update_canvas()

    # def plot_markers_volume(self, **kwargs):
    #     class_markers = self.get_class_markers( **kwargs )
    #     for cid, pids in class_markers.items():
    #         lm().mark_points( np.array(pids), cid )
    #         pcm().update_marked_points( cid )

    def update_canvas(self):
        lgm().log( "update_canvas" )
        self.figure.canvas.draw_idle()

    def mpl_pick_marker( self, event: PickEvent ):
        rightButton: bool = event.mouseevent.button == MouseButton.RIGHT
        if ( event.name == "pick_event" ) and ( event.artist == self.marker_plot ) and rightButton: #  and ( self.key_mode == Qt.Key_Shift ):
            self.delete_marker( event.mouseevent.ydata, event.mouseevent.xdata )
            self.update_plots()

    def delete_marker(self, y, x ):
        pindex = self.block.coords2pindex( y, x )
        lm().deletePid( pindex )

    def initPlots(self, **kwargs) -> Optional[AxesImage]:
        if self.image is None:
            self.image = self.create_image(**kwargs)
            self.overlay_image = self.create_overlay_image()
            if self.image is not None: self.initMarkersPlot()
        return self.image

    def clearMarkersPlot( self ):
        offsets = np.ma.column_stack([[], []])
        self.marker_plot.set_offsets( offsets )
        self.plot_markers_image()

    def setMarkersVisible(self, is_visible: bool ):
        if self.marker_plot:
            new_alpha = 1.0 if is_visible else 0.0
            self.marker_plot.set_alpha( new_alpha )
            self.update_canvas()

    def initMarkersPlot(self):
        print( "Init Markers Plot")
        self.marker_plot: PathCollection = self.plot_axes.scatter([], [], s=50, zorder=3, alpha=1, picker=True)
        self.marker_plot.set_edgecolor([0, 0, 0])
        self.marker_plot.set_linewidth(2)
        self.figure.canvas.mpl_connect('pick_event', self.mpl_pick_marker)
        self.plot_markers_image()

    def add_slider(self,  **kwargs ):
        if self.slider is None:
            self.slider = PageSlider( self.slider_axes, self.nFrames )
            self.slider_cid = self.slider.on_changed(self._update)

    def wait_for_key_press(self):
        keyboardClick = False
        while keyboardClick != True:
            keyboardClick = plt.waitforbuttonpress()

    def _update( self, val ):
        if self.slider is not None:
            tval = self.slider.val
            self.currentFrame = int( tval )
            lgm().log(f"Slider Update, frame = {self.currentFrame}")
            ufm().show( f"Loading frame {self.currentFrame}", "yellow" )
            self.update_plots()
            ufm().clear()

    def show(self):
        plt.show()

    def __del__(self):
        self.exit()

    def exit(self):
        pass
