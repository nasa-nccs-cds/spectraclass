import ipywidgets as ipw
import pythreejs as p3js
import matplotlib.pyplot as plt
import time, math, os, sys, numpy as np
from spectraclass.gui.spatial.widgets.markers import Marker
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from matplotlib import cm
import numpy.ma as ma
import xarray as xa
from spectraclass.data.spatial.voxels import Voxelizer
from matplotlib.colors import Normalize
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import traitlets as tl
from spectraclass.model.labels import LabelsManager, lm, c2rgb
from spectraclass.model.base import SCSingletonConfigurable

def pcm() -> "PointCloudManager":
    return PointCloudManager.instance()

def asarray( data: Union[np.ndarray,Iterable], dtype  ) -> np.ndarray:
    if isinstance( data, np.ndarray ): return data
    else: return np.array( list(data), dtype = np.dtype(dtype) )

class PointCloudManager(SCSingletonConfigurable):

    color_map = tl.Unicode("gist_rainbow").tag(config=True)

    def __init__( self):
        super(PointCloudManager, self).__init__()
        self._gui = None
        self._xyz: np.ndarray = None
        self.points: p3js.Points = None
        self.marker_points: Dict[int,int] = {}
        self.scene: p3js.Scene = None
        self.renderer: p3js.Renderer = None
        self.raycaster = p3js.Raycaster()
        self.pickers: List[p3js.Picker] = []
        self.control_panel: ipw.DOMWidget = None
        self.size_control: ipw.FloatSlider = None
        self.point_locator: np.ndarray = None
        self._color_values = None
        self.reduced_opacity = 0.111111
        self.standard_opacity = 0.411111
        self.transient_markers = []
        self.scale = 100.0
        self.centroid = (0,0,0)
        key_light = p3js.DirectionalLight( color='white', position=[5*self.scale,0,0], intensity=0.4 )
        self.camera = p3js.PerspectiveCamera( fov=90, aspect=1, position=[3.5*self.scale,0,0], up=[0,0,1], children=[key_light] )
        self.camera.lookAt( self.centroid )
        self.orbit_controls = p3js.OrbitControls( controlling=self.camera )
        self.orbit_controls.target = self.centroid
        self.pick_point: int = -1
        self.scene_controls = {}
        self.opacity_control = None
        self.voxelizer: Voxelizer = None
        self.initialize_points()

    def clear_transients(self):
        for pid in self.transient_markers:
            self.marker_points.pop(pid)
        self.transient_markers = []

    def addMarker(self, marker: Marker ):
        self.clear_transients()
        for pid in marker.pids:
            self.marker_points[pid] = marker.cid
            if marker.cid == 0:
                self.transient_markers.append( pid )

    def deleteMarkers( self, pids: List[int] ):
        for pid in pids:
            self.marker_points.pop( pid )

    @property
    def xyz(self)-> np.ndarray:
        return self._xyz

    @xyz.setter
    def xyz(self, value: np.ndarray):
        self._xyz = value
        self._bounds = []
        self.voxelizer = Voxelizer( value, 0.1*self.scale )
        self.point_locator = value.sum(axis=1)

    def initialize_points(self):
        self._xyz: np.ndarray = self.empty_pointset

    @property
    def empty_pointset(self) -> np.ndarray:
        return np.empty(shape=[0, 3], dtype=np.float)

    @property
    def empty_pids(self) -> np.ndarray:
        return np.empty(shape=[0], dtype=np.int)

    def init_data(self, **kwargs):
        from spectraclass.reduction.embedding import ReductionManager, rm
        from spectraclass.data.base import DataManager
        project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("points")
        reduced_data: xa.DataArray = project_dataset.reduction
        reduced_data.attrs['dsid'] = project_dataset.attrs['dsid']
        lgm().log(f"UMAP init, init data shape = {reduced_data.shape}")
        embedding = rm().umap_init(reduced_data, **kwargs)
        self.xyz = self.normalize(embedding)

    def normalize(self, point_data: np.ndarray):
        return (point_data - point_data.mean()) * (self.scale / point_data.std())

    def getColors( self, **kwargs ):
        from spectraclass.gui.spatial.map import MapManager, mm
        norm: Normalize = kwargs.get('norm')
        cdata  = mm().getPointData( current_frame=True )
        if norm is None:
            vr = mm().get_color_bounds( cdata )
            norm = Normalize( vr['vmin'], vr['vmax'] )
        lgm().log( f"getColors: norm cdata shape = {cdata.shape}, dims={cdata.dims}")
        mapper = plt.cm.ScalarMappable( norm = norm, cmap="jet" )
        colors = mapper.to_rgba( cdata.values )[:, :-1] * 255
        if self.pick_point >= 0:
            colors[ self.pick_point ] = [255.0,255.0,255.0]
        return colors.astype(np.uint8)

    def getGeometry( self, **kwargs ):
        colors = self.getColors( **kwargs )
        lgm().log(f"getColors: xyz shape = {self.xyz.shape}")
        attrs = dict( position = p3js.BufferAttribute( self.xyz, normalized=False ),
                      color =    p3js.BufferAttribute( list(map(tuple, colors))) )
        return p3js.BufferGeometry( attributes=attrs )

    def getMarkerGeometry( self ) -> p3js.BufferGeometry:
        markers = dict( sorted( self.marker_points.items() ) )
        colors = lm().get_rgb_colors( np.array( markers.values() ) ).astype(np.uint8)
        positions = self._xyz[ np.array( markers.keys() ) ]
        attrs = dict( position = p3js.BufferAttribute( positions, normalized=False ),
                      color =    p3js.BufferAttribute( colors ) )
        return p3js.BufferGeometry( attributes=attrs )

    def initPoints(self, **kwargs):
        points_geometry = self.getGeometry( **kwargs )
        points_material = p3js.PointsMaterial( vertexColors='VertexColors', transparent=True )
        self.points = p3js.Points( geometry=points_geometry, material=points_material )
        marker_geometry = self.getMarkerGeometry()
        marker_material = p3js.PointsMaterial( vertexColors='VertexColors', transparent=True )
        self.marker_points = p3js.Points( geometry=marker_geometry, material=marker_material )

    def control_label(self, name: str ) -> ipw.Label:
        toks = name.split(".")
        return ipw.Label( f"{toks[0]} {toks[1]}" )

    def getControlsWidget(self) -> ipw.DOMWidget:
        self.scene_controls['point.material.size'] = ipw.FloatSlider(value=0.02 * self.scale, min=0.0, max=0.05 * self.scale, step=0.0002 * self.scale)
        self.scene_controls['point.material.opacity'] = ipw.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.01)
        self.scene_controls['marker.material.size'] = ipw.FloatSlider(value=0.02 * self.scale, min=0.0,                                                                     max=0.05 * self.scale, step=0.0002 * self.scale)
        self.scene_controls['marker.material.opacity'] = ipw.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.01)
        self.scene_controls['window.scene.background'] = ipw.ColorPicker( value="black" )
        self.link_controls()
        return ipw.VBox( [ ipw.HBox( [ self.control_label(name), ctrl ] ) for name, ctrl in self.scene_controls.items() ] )

    def link_controls(self):
        for name, ctrl in self.scene_controls.items():
            toks = name.split(".")
            object = self.points.material if toks[1] == "material" else self.scene
            ipw.jslink( (ctrl, 'value'), (object, toks[2]) )

    def create_picker(self, points: p3js.Points ):
        picker = p3js.Picker(controlling=points, event='click')
        picker.observe( self.on_pick, names=['point'] )
        return picker

    def _get_gui( self ) -> ipw.DOMWidget:
        self.initPoints()
        self.scene = p3js.Scene( children=[ self.points, self.camera, p3js.AmbientLight(intensity=0.8)  ] )
        self.renderer = p3js.Renderer( scene=self.scene, camera=self.camera, controls=[self.orbit_controls], width=800, height=500 )
        self.pickers = [ self.create_picker(points) for points in [ self.points, self.marker_points ] ]
        self.renderer.controls = self.renderer.controls + self.pickers
        self.control_panel = self.getControlsWidget()
        return ipw.VBox( [ self.renderer, self.control_panel ]  )

    @exception_handled
    def on_pick( self, event: Dict ):
        from spectraclass.gui.spatial.map import MapManager, mm
        point = tuple( event["new"] )
        self.pick_point = self.voxelizer.get_pid( point )
        pos = mm().mark_point( self.pick_point, cid=0 )
        lgm().log( f"\n -----> on_pick: pid={self.pick_point}, pos = {pos} [{point}]")
        self.points.geometry = self.getGeometry()

    def gui(self, **kwargs ) -> ipw.DOMWidget:
        if self._gui is None:
            self.init_data( **kwargs )
            self._gui = self._get_gui()
        return self._gui

    def get_index_from_point(self, point: List[float] ) -> int:
        spt = sum(point)
        loc_array: np.ndarray = np.abs( self.point_locator - spt )
        indx = np.argmin( loc_array )
        lgm().log( f"get_index_from_point[{indx}]: Loc array range=[{loc_array.min()},{loc_array.max()}], spt={loc_array[indx]}, pos={self.xyz[indx]}")
        return indx

    def update_plot(self, **kwargs):
        if 'points' in kwargs:
            self.xyz = self.normalize(kwargs['points'])
        if self._gui is not None:
            lgm().log( " *** update point cloud data *** " )
            self.points.geometry = self.getGeometry()

    def clear(self):
        self.update_plot()

    def color_by_value(self, values: np.ndarray = None, **kwargs):
        is_distance = kwargs.get('distance', False)
        if values is not None:
            lgm().log(f" $$$color_by_value[distance:{is_distance}]: data shape = {values.shape} ")
            self._color_values = ma.masked_invalid(values)
        if self._color_values is not None:
            colors = self._color_values.filled(-1)
            (vmin, vmax), npb = ((0.0,
                                  self._color_values.max()) if is_distance else self.get_color_bounds()), self._n_point_bins
            lgm().log(
                f" $$$color_by_value(shape:{colors.shape}) (vmin, vmax, npb) = {(vmin, vmax, npb)}, points (max, min, shape) = {(self._points.max(), self._points.min(), self._points.shape)}")
            pts: np.ndarray = ma.masked_invalid(self._points).filled(-1)
            lspace: np.ndarray = np.linspace(vmin, vmax, npb + 1)
            for iC in range(0, npb):
                if iC == 0:
                    mask = colors <= lspace[iC + 1]
                elif (iC == npb - 1):
                    mask = (colors > lspace[iC]) & (colors < sys.float_info.max)
                else:
                    mask = (colors > lspace[iC]) & (colors <= lspace[iC + 1])
#                self._binned_points[iC] = pts[mask]
            #                lgm().log(f" $$$COLOR: BIN-{iC}, [ {lspace[iC]} -> {lspace[iC+1]} ], nvals = {self._binned_points[iC].shape[0]}, #mask-points = {np.count_nonzero(mask)}" )
            self.set_base_points_alpha(self.reduced_opacity)
            self.update_plot(**kwargs)

    def get_color_bounds(self):
        from spectraclass.data.spatial.manager import SpatialDataManager
        (ave, std) = (self._color_values.mean(), self._color_values.std())
        return (ave - std * SpatialDataManager.colorstretch, ave + std * SpatialDataManager.colorstretch)

    def set_base_points_alpha( self, alpha: float ):
        alphas = list( self._gui.point_set_opacities )
        alphas[0] = alpha
        self._gui.point_set_opacities = alphas
        lgm().log(f"Set point set opacities: {self._gui.point_set_opacities}")
        self.update_plot( alphas = alphas )

    def refresh(self):
        self.initialize_points()
        self.init_data()
        self.update_plot()


