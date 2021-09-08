
from typing import List, Union, Tuple, Optional, Dict, Callable
from collections import OrderedDict
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.model.base import SCSingletonConfigurable, Marker
from spectraclass.gui.spatial.widgets.events import em
from functools import partial
import traitlets as tl
import ipywidgets as ipw
import rioxarray as rio
from shapely.geometry.base import BaseGeometry
from rioxarray.raster_dataset import RasterDataset
from shapely import geometry
from spectraclass.gui.spatial.widgets.tools import PageSlider
from spectraclass.gui.spatial.widgets.selection import *
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.util.logs import LogManager, lgm, exception_handled
from spectraclass.gui.spatial.widgets.layers import LayersManager
import types, pandas as pd
import xarray as xa
import numpy as np
import math, atexit, os, traceback
import pathlib
from spectraclass.gui.spatial.widgets.controls import am, ufm
from  ipympl.backend_nbagg import Toolbar
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.colors import Normalize
from matplotlib.backend_bases import PickEvent, MouseButton
from spectraclass.gui.control import UserFeedbackManager, ufm

class RegionTypes:
    Polygon = "Polygon"
    Rectangle = "Rectangle"
    Lasso = "Lasso"
    Point = "Point"
    All = [ Polygon, Rectangle, Lasso, Point ]

def get_color_bounds( color_values: List[float] ) -> List[float]:
    color_bounds = []
    for iC, cval in enumerate( color_values ):
        if iC == 0: color_bounds.append( cval - 0.5 )
        else: color_bounds.append( (cval + color_values[iC-1])/2.0 )
    color_bounds.append( color_values[-1] + 0.5 )
    return color_bounds

def tss() -> "TrainingSetSelection":
    is_initialized = TrainingSetSelection.initialized()
    mgr = TrainingSetSelection.instance()
    if not is_initialized: mgr.observe( mgr.on_overlay_alpha_change, names=["overlay_alpha"] )
    return mgr

class TrainingSetSelection(SCSingletonConfigurable):
    init_band = tl.Int(10).tag( config=True, sync=True )
    overlay_alpha = tl.Float(0.5).tag( config=True, sync=True )

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1

    def __init__( self, **kwargs ):   # class_labels: [ [label, RGBA] ... ]
        super(TrainingSetSelection, self).__init__()
        self._debug = False
        self.currentFrame = 0
        self.block: Block = None
        self.slider: Optional[PageSlider] = None
        self.image: Optional[AxesImage] = None
        self.image_template: Optional[xa.DataArray]  = None
        self.overlay_image: Optional[AxesImage] = None
#        self._classification_data: Optional[np.ndarray] = None
        self._layer_manager: LayersManager = None
        self._selection_tool: SelectionTool = None
        self.use_model_data: bool = False
        self._label_files_stack: List[str] = []
        self._labels_map: xa.DataArray = None
        self.transients = []
        self.plot_axes: Optional[Axes] = None
        self.marker_plot: Optional[PathCollection] = None
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
        self._control_panels = OrderedDict()
        self.setup_plot(**kwargs)

#        google_actions = [[maptype, None, None, partial(self.run_task, self.download_google_map, "Accessing Landsat Image...", maptype, task_context='newfig')] for maptype in ['satellite', 'hybrid', 'terrain', 'roadmap']]
        self.menu_actions = OrderedDict( Layers = [ [ "Increase Labels Alpha", 'Ctrl+>', None, partial( self.update_image_alpha, "labels", True ) ],
                                                    [ "Decrease Labels Alpha", 'Ctrl+<', None, partial( self.update_image_alpha, "labels", False ) ],
                                                    [ "Increase Band Alpha",   'Alt+>',  None, partial( self.update_image_alpha, "bands", True ) ],
                                                    [ "Decrease Band Alpha",   'Alt+<',  None, partial( self.update_image_alpha, "bands", False ) ] ] )

        atexit.register(self.exit)
        self._update(0)

    @property
    def region_type(self) -> str:
        return str( self._region_types.value )

    def select_region( self, *args, **kwargs ):
        ufm().show( f"select_region, type = {self.region_type}" )
        if self.region_type == RegionTypes.Polygon: Selector = PolygonSelectionTool
        elif self.region_type == RegionTypes.Point: Selector = PointSelectionTool
        elif self.region_type == RegionTypes.Rectangle: Selector = RectangleSelectionTool
        elif self.region_type == RegionTypes.Lasso: Selector = LassoSelectionTool
        else: raise NotImplementedError( f"Select tool not implemented for {self.region_type}")
        self.clear_selection()
        self._selection_tool = Selector( self.figure    )
        self._selection_tool.enable()

    def clear_selection(self):
        if self._selection_tool is not None:
            self._selection_tool.disable()
            self._selection_tool = None

    def label_region( self, *args, **kwargs ):
        iClass: int = lm().current_cid
        regions: List[BaseGeometry] = self.getSelectedRegions()
        self._labels_map = self.fill_regions( self._labels_map, regions, iClass )
        self.display_label_image()
        self.clear_selection()
        lm().setLabelData( self._labels_map.data )
#        self.save_labels( self._labels_map )

    def display_label_image(self):
        self.overlay_image.set_data( self._labels_map )
        self.overlay_image.set_alpha( self.overlay_alpha )
        self._layer_manager.layer("labels").update( self.overlay_alpha, True )
        self.update_canvas()

    def fill_regions(self, data_array: xa.DataArray, regions: List[BaseGeometry], fill_value: int ):
        from rasterio.mask import  geometry_mask
        from affine import Affine
        mask = geometry_mask( regions, transform=Affine( *data_array.transform[:6] ), out_shape=data_array.shape, all_touched=True )
        return data_array.where( mask, fill_value )

    def addPanel(self, name: str, widget: ipw.Widget ):
        self._control_panels[ name ] = widget

    def getLearningPanel(self):
        return ipw.VBox( [ ] )

    def getLabelsPanel(self):
        self._region_types = ipw.Dropdown( options=RegionTypes.All, description="Region Type", index=0 )
        return ipw.VBox( [ self._region_types ] )

    def getLayersPanel(self):
        if self._layer_manager is None:
            self._layer_manager = LayersManager( self.figure, self.set_layer_alpha )
        self._layer_manager.add_layer( "data", 1.0, True )
        self._layer_manager.add_layer( "labels", 0.5, True )
        return self._layer_manager.gui()

    def set_layer_alpha( self, name: str, alpha: float ):
        if name == "data":      self.image.set_alpha( alpha )
        elif name == "labels":  self.overlay_image.set_alpha( alpha )
        else: raise Exception( f"Unrecognized layer name: {name}" )
        self.update_canvas()

    def getControlPanel(self):
        control_collapsibles = ipw.Accordion( children=tuple(self._control_panels.values()), layout=ipw.Layout(width='300px'))  #
        for iT, title in enumerate(self._control_panels.keys()): control_collapsibles.set_title(iT, title)
        control_collapsibles.selected_index = 1
        return control_collapsibles

    def gui(self):
        css_border = '1px solid blue'
        self.setBlock()
        self.defineActions()
        top_layout = ipw.Layout( width="100%", height="120px",  border= '2px solid firebrick' ) # justify_content="space-between", flex='0 0 70px',
        actions = ipw.VBox([ ufm().gui(), lm().gui(), am().gui() ], layout = top_layout )
        map_panels = ipw.HBox( [ self.figure.canvas, self.getControlPanel() ], border=css_border )
        tsgui = ipw.VBox( [ actions, map_panels ], border=css_border )
        return tsgui

    def defineActions(self):
        am().add_action( "select",  self.select_region )
        am().add_action( "label",   self.label_region  )
        am().add_action( "augment", self.augment_selection_action)
        am().add_action( "undo",    self.undo_action   )
        am().add_action( "clear",   self.clear_action  )

    def getSelectedRegions(self) -> List[BaseGeometry]:
        assert self._selection_tool is not None, "Must selected a region first."
        selected_vertices: List[Tuple] = self._selection_tool.selection()
        lgm().log(f"Selected Region: {selected_vertices}")
        if self.region_type == RegionTypes.Point:
            return [ geometry.Point(*vertex) for vertex in selected_vertices ]
        else:
            return [ geometry.Polygon( selected_vertices ) ]

    def augment_selection_action(self):
        ufm().show( "mark selection" )

    def undo_action(self):
        ufm().show( "undo" )
        self._labels_map = self._labels_map.copy( data = lm().undoLabelsUpdate() )
        self.display_label_image()

    def clear_action(self):
        ufm().show( "clear" )
        self._labels_map = self._labels_map.copy( data = lm().clearLabels() )
        self.display_label_image()

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

    # def create_mask( self, cid: int ):
    #     from spectraclass.data.base import DataManager, dm
    #     if self._classification_data is None:
    #         ufm().show( "Must generate a classification before creating a mask", "red" )
    #     elif cid == 0:
    #         ufm().show( "Must choose a class in order to create a mask", "red" )
    #     else:
    #         data: xa.DataArray = self.block.data
    #         mask_data: np.ndarray = np.equal( self._classification_data, np.array(cid).reshape((1,1)) )
    #         mask_array = xa.DataArray( mask_data, name=f"mask-{cid}", dims=data.dims[1:], coords= { d:data.coords[d] for d in data.dims[1:] } )
    #         output_file = dm().mask_file
    #         if os.path.exists( output_file ):
    #             mask_dset: xa.Dataset = xa.open_dataset( output_file )
    #             mask_dset.update( { mask_array.name: mask_array } )
    #             mask_dset.to_netcdf( output_file, format='NETCDF4', engine='netcdf4' )
    #         else:
    #             mask_array.to_netcdf( output_file, format='NETCDF4', engine='netcdf4' )
    #         lgm().log( f"\n\n ###### create mask: {mask_array} \n Saved to file: {output_file}" )
    #
    # @exception_handled
    # def plot_overlay_image( self, image_data: np.ndarray = None ):
    #     if image_data is not None:
    #         lgm().log( f" plot image overlay, shape = {image_data.shape}, vrange = {[ image_data.min(), image_data.max() ]}, dtype = {image_data.dtype}" )
    #         self._classification_data = image_data
    #         self.overlay_image.set_data( image_data )
    #     self.overlay_image.set_alpha( self.overlay_alpha )
    #     self.update_canvas()

    def on_overlay_alpha_change(self, *args ):
        self.overlay_image.set_alpha( self.overlay_alpha )
        lgm().log(f" image overlay set alpha = {self.overlay_alpha}" )
        self.update_canvas()

    def setBlock( self, **kwargs ) -> Block:
        from spectraclass.data.spatial.tile.manager import TileManager
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
                self.add_tools()
                self.add_slider(**kwargs)
                self.initLabels()
                self.update_plot_axis_bounds()
                self.update_plots()

        return self.block

    def add_tools(self):
        pass

    def update_plot_axis_bounds( self ):
        if self.plot_axes is not None:
            self.plot_axes.set_xlim( self.block.xlim )
            self.plot_axes.set_ylim( self.block.ylim )

    def initLabels(self):
        nodata_value = -2
        self._labels_map: xa.DataArray = self.image_template.copy( data=np.full( self.image_template.shape, -1, np.int32 ) )
        self._labels_map.attrs['_FillValue'] = nodata_value
        self._labels_map.name = "labels"
        self._labels_map.attrs[ 'long_name' ] = [ "labels" ]
        self._labels_map = self._labels_map.where( self.image_template.notnull(), nodata_value )
        lm().setLabelData( self._labels_map.data )
#        self.save_labels( self._labels_map )

    def save_labels(self, labels: xa.DataArray ):
        from datetime import datetime
        now = datetime.now() # current date and time
        date_time = now.strftime("%m.%d.%Y_%H.%M.%S")
        dset_name = f"{self.block.data.name}_labels_{date_time}"
        file_name = self.writeGeotiff( dset_name, labels )
        if file_name is not None:
            self._label_files_stack.append( dset_name )

    def get_raster_file_name(self, dset_name: str ) -> str:
        from spectraclass.data.base import DataManager, dm
        return f"{dm().modal.data_dir}/{dset_name}.tif"

    def read_labels( self ) -> xa.Dataset:
        dset_name = self._label_files_stack[-1]
        labels_file = self.get_raster_file_name(dset_name)
        return rio.open_rasterio(labels_file)

    def writeGeotiff(self, dset_name: str, raster_data: xa.DataArray ) -> Optional[str]:
        output_file = self.get_raster_file_name(dset_name)
        try:
            if os.path.exists(output_file): os.remove(output_file)
            lgm().log(f"Writing (raster) tile file {output_file}")
            raster_data.rio.to_raster(output_file)
            return output_file
        except Exception as err:
            lgm().log(f"Unable to write raster file to {output_file}: {err}")
            return None

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

    def onPolySelect(self, vertices: List ):
        lgm().log(f"vertices selected = {vertices}")

    def setup_plot(self, **kwargs):
        self.figure.suptitle("Band Image")
        self.plot_axes:   Axes = self.figure.add_axes([0.01, 0.07, 0.98, 0.93])  # [left, bottom, width, height]
        self.plot_axes.xaxis.set_visible( False ); self.plot_axes.yaxis.set_visible( False )
        self.slider_axes: Axes = self.figure.add_axes([0.01, 0.01, 0.85, 0.05])  # [left, bottom, width, height]
        self.figure.canvas.toolbar_visible = True
        self.figure.canvas.header_visible = False
        lgm().log( f"Canvas class = {self.figure.canvas.__class__}" )
        lgm().log( f"Canvas.manager class = {self.figure.canvas.manager.__class__}")
        items = self.figure.canvas.trait_values().items()
        for k,v in items: lgm().log(f" ** {k}: {v}")
        self._control_panels['Selection'] = self.getLabelsPanel()
        self._control_panels['Layers'] = self.getLayersPanel()
        self._control_panels['Augment'] = self.getLearningPanel()

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
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.manager import SpatialDataManager
        dms: SpatialDataManager = dm().modal
        self.image_template: xa.DataArray =  self.data[ self.init_band, :, : ]
        nValid = np.count_nonzero(~np.isnan(self.image_template))
        lgm().log( f"\n ********* Creating Map Image, nValid={nValid}, data shape = {self.data.shape}, image shape = {self.image_template.shape}, band = {self.init_band}, data range = [ {self.data.min().values}, {self.data.max().values} ]")
        assert nValid > 0, "No valid pixels in image"
        colorbar = kwargs.pop( 'colorbar', False )
        image: AxesImage =  dms.plotRaster( self.image_template, ax=self.plot_axes, colorbar=colorbar, alpha=0.5, **kwargs )
        self._cidpress = image.figure.canvas.mpl_connect('button_press_event', self.onMouseClick)
        self._cidrelease = image.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease )
#        lgm().log( f"TOOLBAR: {image.figure.canvas.manager.toolbar.__class__}" )
#        image.figure.canvas.manager.toolmanager.add_tool("ToggleSource", ToggleDataSourceMode)
#        image.figure.canvas.manager.toolbar.add_tool("ToggleSource", 'navigation', 1)
        self.plot_axes.callbacks.connect('ylim_changed', self.on_lims_change)
        overlays = kwargs.get( "overlays", {} )
        for color, overlay in overlays.items():
            overlay.plot( ax=self.plot_axes, color=color, linewidth=2 )
        return image

    @exception_handled
    def create_overlay_image( self ) -> AxesImage:
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.manager import SpatialDataManager
        dms: SpatialDataManager = dm().modal
        assert self.image is not None, "Must create base image before overlay"
        overlay_image: AxesImage =  dms.plotRaster( self.image_template, itype='overlay', colorbar=False, alpha=0.0, ax=self.plot_axes, zeros=True )
        return overlay_image

    def on_lims_change(self, ax ):
         if ax == self.plot_axes:
             (x0, x1) = ax.get_xlim()
             (y0, y1) = ax.get_ylim()
             print(f"ZOOM Event: Updated bounds: ({x0},{x1}), ({y0},{y1})")

    def get_frame_data( self, **kwargs ) -> Optional[xa.DataArray]:
        if self.currentFrame >= self.data.shape[0]: return None
        frame_data: xa.DataArray = self.data[ self.currentFrame ]
        return frame_data

    @exception_handled
    def update_plots(self):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.manager import SpatialDataManager
        dms: SpatialDataManager = dm().modal
        if self.image is not None:
            frame_data: xa.DataArray = self.get_frame_data()
            if frame_data is not None:
                self.image.set_data( frame_data.values  )
                drange = dms.get_color_bounds( frame_data )
                self.image.set_norm( Normalize( **drange ) )
                self.image.set_extent( self.block.extent() )
                plot_name = os.path.basename( dm().dsid() )
                lgm().log( f" Update Map: data shape = {frame_data.shape}, range = {drange}, extent = {self.block.extent()}")
                self.plot_axes.title.set_text(f"{plot_name}: Band {self.currentFrame+1}" )
                self.plot_axes.title.set_fontsize( 8 )
                self.update_canvas()

    def clear_overlay_image(self):
        self.overlay_image.set_extent(self.block.extent())
        self.overlay_image.set_alpha(0.0)
        self.update_canvas()

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
                    # lgm().log( f"Adding marker for pid = {pid}, cid = {cid}")
                    # ptindices = self.block.pindex2indices(pid)
                    # classification = self.label_map.values[ ptindices['iy'], ptindices['ix'] ] if (self.label_map is not None) else -1
                    # self.add_marker( Marker( [pid], cid, classification = classification ) )
                    # self.dataLims = event.inaxes.dataLim
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
        if not self._adding_marker:
            self._adding_marker = True
            if marker is None:
                lgm().log( "NULL Marker: point select is probably out of bounds.")
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
    #         labels_map: xa.DataArray = self.getLabeledPointData()
    #         sample_labels: Optional[xa.DataArray] = self.block.flow.spread( labels_map, self.flow_iterations, **kwargs )
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

    # def show_labels(self):
    #     if self.labels_image is not None:
    #         self.labels_image.set_alpha(1.0)
    #         self.update_canvas()
    #
    # def toggle_labels(self):
    #     if self.labels_image is not None:
    #         new_alpha = 1.0 if (self.labels_image.get_alpha() == 0.0) else 0.0
    #         self.labels_image.set_alpha( new_alpha )
    #         self.update_canvas()

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

    # def get_markers( self ) -> Tuple[ List[float], List[float], List[List[float]] ]:
    #     ycoords, xcoords, colors, markers = [], [], [], lm().markers
    #     lgm().log(f" ** get_markers, #markers = {len(markers)}")
    #     for marker in markers:
    #         for pid in marker.pids:
    #             coords = self.block.pindex2coords( pid )
    #             if (coords is not None) and self.block.inBounds( coords['y'], coords['x'] ):   #  and not ( labeled and (c==0) ):
    #                 ycoords.append( coords['y'] )
    #                 xcoords.append( coords['x'] )
    #                 colors.append( lm().colors[marker.cid] )
    #             else:
    #                 lgm().log(f" ** coords[{pid}] out of bounds: {[coords['y'], coords['x']]}, bounds = ( {self.block._ylim}, {self.block._xlim} )")
    #                 lgm().log(f" ** Point coords range: {[coords['y'], coords['x']]}")
    #                 lgm().log(f" ** Projection bounds: xlim = {self.block.xlim}, ylim = {self.block.ylim} " )
    #                 yc = self.block.point_coords['y']; xc = self.block.point_coords['x']
    #                 lgm().log(f" ** Coordinates bounds: xrange = {[xc.min(),xc.max()]}, yrange = {[yc.min(),yc.max()]} ")
    #     return ycoords, xcoords, colors

    # def get_class_markers( self, **kwargs ) -> Dict[ int, List[int] ]:
    #     class_markers = {}
    #     for marker in lm().getMarkers():
    #         pids = class_markers.setdefault( marker.cid, [] )
    #         pids.extend( marker.pids )
    #     return class_markers

    # @exception_handled
    # def plot_markers_image( self ):
    #     if self.marker_plot:
    #         ycoords, xcoords, colors = self.get_markers()
    #         lgm().log(f" ** plot markers image, nmarkers = {len(ycoords)}")
    #         if len(ycoords) > 0:
    #             self.marker_plot.set_offsets(np.c_[xcoords, ycoords])
    #             self.marker_plot.set_facecolor(colors)
    #         else:
    #             offsets = np.ma.column_stack([[], []])
    #             self.marker_plot.set_offsets(offsets)
    #         self.update_canvas()

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
        return self.image

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
