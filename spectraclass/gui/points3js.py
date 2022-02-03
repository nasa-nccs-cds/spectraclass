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
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.model.base import SCSingletonConfigurable

def pcm() -> "PointCloudManager":
    return PointCloudManager.instance()

def asarray( data: Union[np.ndarray,Iterable], dtype  ) -> np.ndarray:
    if isinstance( data, np.ndarray ): return data
    else: return np.array( list(data), dtype = np.dtype(dtype) )

class PointCloudManager(SCSingletonConfigurable):

    color_map = tl.Unicode("gist_rainbow").tag(config=True)  # "gist_rainbow" "jet"

    def __init__( self):
        super(PointCloudManager, self).__init__()
        self._gui = None
        self._xyz: np.ndarray = None
        self.points: p3js.Points = None
        self.scene: p3js.Scene = None
        self.renderer: p3js.Renderer = None
        self.raycaster = p3js.Raycaster()
        self.picker: p3js.Picker = None
        self.control_panel: ipw.DOMWidget = None
        self.size_control: ipw.FloatSlider = None
        self.point_locator: np.ndarray = None
        self._color_values = None
        self.reduced_opacity = 0.111111
        self.standard_opacity = 0.411111
        self.scale = 100.0
        self.centroid = (0,0,0)
        key_light = p3js.DirectionalLight( color='white', position=[5*self.scale,0,0], intensity=0.4 )
        self.camera = p3js.PerspectiveCamera( fov=90, aspect=1, position=[3.5*self.scale,0,0], up=[0,0,1], children=[key_light] )
        self.camera.lookAt( self.centroid )
        self.orbit_controls = p3js.OrbitControls( controlling=self.camera )
        self.orbit_controls.target = self.centroid
        self.pick_point: int = -1
        self.marker_points: Dict[int,p3js.Mesh] = {}
        self.voxelizer: Voxelizer = None
        self.initialize_points()

    def addMarker(self, marker: Marker ):
        for pid in marker.pids:
            if pid not in self.marker_points:
                material = p3js.MeshLambertMaterial( color= lm().colors[ marker.cid ] )
                geometry = p3js.SphereGeometry( radius= 0.01*self.scale )
                marker_point = p3js.Mesh( geometry=geometry, material=material )
                marker_point.position = tuple( self.xyz[ pid ].tolist() )
                self.scene.add(marker_point)
                self.marker_points[pid] = marker_point

    def deleteMarker( self, pid: int ):
        marker_point = self.marker_points.pop(pid)
        if marker_point is not None:
            self.scene.remove( marker_point )

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
                      color =    p3js.BufferAttribute( list(map(tuple, colors))),
                      alpha =    kwargs.get('alpha',1.0) )
        return p3js.BufferGeometry( attributes=attrs )

    def getPoints( self, **kwargs ) -> p3js.Points:
        points_geometry = self.getGeometry( **kwargs )
        points_material = p3js.PointsMaterial( vertexColors='VertexColors')
        points = p3js.Points( geometry=points_geometry, material=points_material )
        if self.size_control is not None:
            ipw.jslink( (self.size_control, 'value'), ( points_material, 'size' ) )
        if self.picker is not None:
            self.picker.controlling = points
        return points

    def getControlsWidget(self) -> ipw.DOMWidget:
        self.size_control = ipw.FloatSlider( value=0.02*self.scale, min=0.0, max=0.05*self.scale, step=0.0002*self.scale )
        self.opacity_control = ipw.FloatSlider( value=1.0, min=0.0, max=1.0, step=0.01 )
        ipw.jslink( (self.size_control,'value'), (self.points.material,'size') )
#        ipw.jslink( (self.opacity_control, 'value'), (self.points.material, 'alpha'))
        color = ipw.ColorPicker( value="black" )
        ipw.jslink( (color,'value'), (self.scene,'background') )
        psw = ipw.HBox( [ ipw.Label('Point size:'), self.size_control ] )
        pow = ipw.HBox([ipw.Label('Point opacity:'), self.opacity_control ] )
        bcw = ipw.HBox( [ ipw.Label('Background color:'), color ] )
        return ipw.VBox( [ psw, pow, bcw ] )

    def _get_gui( self ) -> ipw.DOMWidget:
        self.points = self.getPoints()
        self.scene = p3js.Scene( children=[ self.points, self.camera, p3js.AmbientLight(intensity=0.8)  ] )
        self.renderer = p3js.Renderer( scene=self.scene, camera=self.camera, controls=[self.orbit_controls], width=800, height=500 )
        self.picker = p3js.Picker( controlling=self.points, event='click')
        self.picker.all = False
        self.picker.observe( self.on_pick, names=['point'] )
        self.renderer.controls = self.renderer.controls + [ self.picker ]
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

#            if 'alphas' in kwargs: self._gui.point_set_opacities = kwargs['alphas']
#            self._gui.update_rendered_image()

    def on_selection(self, selection_event: Dict):
        selection = selection_event['pids']
        self.update_markers(selection)
        self.update_plot()

    def update_markers(self, pids: List[int] = None, **kwargs):
        if pids is None:
            if 'points' in kwargs: self._points = self.normalize(kwargs['points'])
            self.initialize_markers(True)
            for marker in lm().markers:
                self._marker_pids[marker.cid] = np.append(self._marker_pids[marker.cid], marker.pids, 0)
                self._marker_points[marker.cid] = self._points[self._marker_pids[marker.cid], :]
        else:
            self._marker_pids[0] = np.array(pids)
            self._marker_points[0] = self._points[self._marker_pids[0], :]
            lgm().log(f"  ***** POINTS- mark_points[0], #pids = {len(pids)}")

    @exception_handled
    def update_marked_points(self, cid: int = -1, **kwargs):
        return
        # from spectraclass.gui.control import UserFeedbackManager, ufm
        # if self._points is None:
        #     ufm().show("Can't mark points in PointCloudManager which is not initialized", "red")
        # else:
        #     icid = cid if cid >= 0 else lm().current_cid
        #     self.initialize_markers()
        #     self.clear_points(0)
        #     self._marker_pids[icid] = asarray(kwargs.get('pids', lm().getPids(icid)), np.int)
        #     self._marker_points[icid] = self._points[self._marker_pids[icid], :]
        #     lgm().log(
        #         f"PointCloudManager.add_marked_points: cid = {icid}, #marked-points[cid]: [{self._marker_pids[icid].size}]")
        # self.set_base_points_alpha(self.reduced_opacity)
        # self.update_plot()

    def clear(self):
        self.initialize_markers(True)
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

    @exception_handled
    def clear_pids(self, cid: int, pids: np.ndarray, **kwargs):
        lgm().log(f"PCM.clear_pids: marker_pids = {[p.tolist() for p in self._marker_pids]}")
        if self._marker_pids is not None:
            dpts = np.vectorize(lambda x: x in pids)
            for iC, marker_pids in enumerate(self._marker_pids):
                if (cid < 0) or (iC == cid):
                    if len(marker_pids) > 0:
                        self._marker_pids[iC] = np.delete(self._marker_pids[iC], dpts(marker_pids))
                        self._marker_points[iC] = self._points[self._marker_pids[iC], :] if len(
                            self._marker_pids[iC]) > 0 else self.empty_pointset
                        lgm().log(
                            f"  --> cid={cid}, # pids cleared = {pids.size}, # remaining markers = {len(self._marker_pids[iC])}")
                    else:
                        self._marker_points[iC] = self.empty_pointset

    def clear_points(self, icid: int, **kwargs):
        if self._marker_pids is not None:
            pids = kwargs.get('pids', None)
            if pids is None:
                lgm().log(f"PCM.clear_points[{icid}]: empty_pointset")
                self._marker_points[icid] = self.empty_pointset
                self._marker_pids[icid] = self.empty_pids
            else:
                lgm().log(f"PCM.clear_points: cid={icid}, #pids={pids.size}")
                dpts = np.vectorize(lambda x: x in pids)
                dmask = dpts(self._marker_pids[icid])
                #            lgm().log( f"clear_points.Mask: {self._marker_pids[icid]} -> {dmask}" )
                self._marker_pids[icid] = np.delete(self._marker_pids[icid], dmask)
                self._marker_points[icid] = self._points[self._marker_pids[icid], :] if len(
                    self._marker_pids[icid]) > 0 else self.empty_pointset

    #            lgm().log(f"clear_points: reduced marker_pids = {self._marker_pids[icid]} -> points = {self._marker_points[ icid ]}")

    def set_base_points_alpha( self, alpha: float ):
        alphas = list( self._gui.point_set_opacities )
        alphas[0] = alpha
        self._gui.point_set_opacities = alphas
        lgm().log(f"Set point set opacities: {self._gui.point_set_opacities}")
        self.update_plot( alphas = alphas )

    def refresh(self):
        self.initialize_points()
        self.initialize_markers( True )
        self.init_data()
        self.update_plot()


