import ipywidgets as ipw
import pythreejs as p3js
import matplotlib.pyplot as plt
import time, math, os, sys, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from matplotlib import cm
import numpy.ma as ma
import xarray as xa
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
        self.xyz: np.ndarray = None
        self.points: p3js.Points = None
        self.scene: p3js.Scene = None
        self.renderer: p3js.Renderer = None
        self.picker: p3js.Picker = None
        self.control_panel: ipw.DOMWidget = None
        self.size_control: ipw.FloatSlider = None
        self._color_values = None
        self.reduced_opacity = 0.111111
        self.standard_opacity = 0.411111
        self.centroid = (0,0,0)
        self.camera = p3js.PerspectiveCamera( fov=90, aspect=1, position=[5,5,1], up=[0,0,1] )
        self.camera.lookAt( self.centroid )
        self.orbit_controls = p3js.OrbitControls( controlling=self.camera )
        self.orbit_controls.target = self.centroid
        self.initialize_points()

    def initialize_points(self):
        self.xyz: np.ndarray = self.empty_pointset
        self._marker_points: List[np.ndarray] = None
        self._marker_pids: List[np.ndarray] = None
        self._color_values = None

    def initialize_markers(self, reset=False):
        if (self._marker_points is None) or reset:
            nLabels = lm().nLabels
            self._marker_points: List[np.ndarray] = [self.empty_pointset for ic in range(nLabels)]
            self._marker_pids: List[np.ndarray] = [self.empty_pids for ic in range(nLabels)]
            lgm().log(f"PCM.initialize_markers, sizes = {[m.size for m in self._marker_points]}")

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
        self.initialize_markers()

    def normalize(self, point_data: np.ndarray):
        return (point_data - point_data.mean()) / point_data.std()

    def getColors( self, **kwargs ):
        from spectraclass.gui.spatial.map import MapManager, mm
        norm: Normalize = kwargs.get('norm')
        cdata = kwargs.get( 'cdata', mm().frame_data )
        if norm is None:
            vr = mm().get_color_bounds( cdata )
            norm = Normalize( vr['vmin'], vr['vmax'] )
        mapper = plt.cm.ScalarMappable( norm = norm, cmap="jet" )
        colors = mapper.to_rgba( cdata.values )[:, :-1] * 255
        return colors.astype(np.uint8)

    def getPoints( self, **kwargs ) -> p3js.Points:
        colors = self.getColors( **kwargs )
        attrs = dict( position=p3js.BufferAttribute( self.xyz, normalized=False ),
                      color=p3js.BufferAttribute(list(map(tuple, colors))))
        points_geometry = p3js.BufferGeometry( attributes=attrs )
        points_material = p3js.PointsMaterial( vertexColors='VertexColors')
        points = p3js.Points( geometry=points_geometry, material=points_material)
        if self.size_control is not None:
            ipw.jslink( (self.size_control, 'value'), ( points_material, 'size' ) )
        if self.picker is not None:
            self.picker.controlling = points
        return points

    def getControlsWidget(self) -> ipw.DOMWidget:
        self.size_control = ipw.FloatSlider( value=0.02, min=0.0, max=0.05, step=0.0002 )
        ipw.jslink( (self.size_control,'value'), (self.points.material,'size') )
        color = ipw.ColorPicker( value="black" )
        ipw.jslink( (color,'value'), (self.scene,'background') )
        psw = ipw.HBox( [ ipw.Label('Point size:'), self.size_control ] )
        bcw = ipw.HBox( [ ipw.Label('Background color:'), color ] )
        return ipw.VBox( [ psw, bcw ] )

    def _get_gui( self ) -> ipw.DOMWidget:
        self.points = self.getPoints()
        self.scene = p3js.Scene( children=[ self.points, self.camera ] )
        self.renderer = p3js.Renderer( scene=self.scene, camera=self.camera, controls=[self.orbit_controls], width=800, height=500 )
        self.picker = p3js.Picker( controlling=self.points, event='click')
        self.picker.observe( self.on_pick, names=['point'])
        self.control_panel = self.getControlsWidget()
        return ipw.VBox( [ self.renderer, self.control_panel ]  )

    def on_pick( self, event ):
        lgm().log( f"on_pick: {event}" )

    def gui(self, **kwargs ) -> ipw.DOMWidget:
        if self._gui is None:
            self.init_data( **kwargs )
            self._gui = self._get_gui()
        return self._gui

    def reembed(self, embedding):
        self.update_plot( points=embedding )

    def update_plot(self, **kwargs):
        if 'points' in kwargs:
            self.xyz = self.normalize(kwargs['points'])
        if self._gui is not None:
            lgm().log( " *** update point cloud data *** " )
            self.scene.remove( self.points )
            self.points = self.getPoints( **kwargs )
            self.scene.add( self.points )

#            if 'alphas' in kwargs: self._gui.point_set_opacities = kwargs['alphas']
#            self._gui.update_rendered_image()

    def on_selection(self, selection_event: Dict):
        selection = selection_event['pids']
        self.update_markers(selection)
        self.update_plot()

    def update_points(self, **kwargs ):
        self.update_plot( **kwargs )

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
        from spectraclass.gui.control import UserFeedbackManager, ufm
        if self._points is None:
            ufm().show("Can't mark points in PointCloudManager which is not initialized", "red")
        else:
            icid = cid if cid >= 0 else lm().current_cid
            self.initialize_markers()
            self.clear_points(0)
            self._marker_pids[icid] = asarray(kwargs.get('pids', lm().getPids(icid)), np.int)
            self._marker_points[icid] = self._points[self._marker_pids[icid], :]
            lgm().log(
                f"PointCloudManager.add_marked_points: cid = {icid}, #marked-points[cid]: [{self._marker_pids[icid].size}]")
        self.set_base_points_alpha(self.reduced_opacity)
        self.update_plot()

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


