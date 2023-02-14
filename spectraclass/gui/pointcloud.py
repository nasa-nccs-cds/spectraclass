import ipywidgets as ipw
import pythreejs as p3js
import matplotlib.pyplot as plt
from functools import partial
import time, math, os, sys, numpy as np
from spectraclass.gui.spatial.widgets.markers import Marker
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from matplotlib import cm
import numpy.ma as ma
import pickle, xarray as xa
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
#    opacity = tl.Float( 'opacity', min=0.0, max=1.0 ).tag(sync=True)

    def __init__( self):
        super(PointCloudManager, self).__init__()
        self._gui = None
        self._xyz: xa.DataArray = None
        self.points: p3js.Points = None
        self.marker_points: Optional[p3js.Points] = None
        self.marker_gids = {}
        self.probe_points: Optional[p3js.Points] = None
        self.probe_gids = {}
        self.scene: p3js.Scene = None
        self.renderer: p3js.Renderer = None
        self.raycaster = p3js.Raycaster()
        self.point_picker: p3js.Picker = None
        self.marker_picker: p3js.Picker = None
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
        self.marker_spheres: Dict[int, p3js.Mesh] = {}
        self.scene_controls = {}
        self.marker_material = p3js.PointsMaterial( vertexColors='VertexColors', transparent=True ) # , size=5.0, opacity=1.0 )
        self.probe_material = p3js.PointsMaterial( vertexColors='VertexColors', transparent=True ) # , size=5.0, opacity=1.0 )
        self.opacity_control = None
        self.voxelizer: Voxelizer = None

    def set_alpha(self, opacity: float ):
        self.scene_controls[ 'marker.material.opacity' ] = opacity

    @exception_handled
    def clear_transients(self):
        self.probe_gids = {}

    @exception_handled
    def mark_point( self, gid: int, cid: int ):
        if cid == 0:    self.probe_gids[gid] = cid
        else:           self.marker_gids[gid] = cid
        return gid

    @exception_handled
    def unmark_point( self, gid: int ) -> int:
        marked_pid = self.marker_gids.pop(gid, -1)
        if marked_pid == -1:
            marked_pid = self.probe_gids.pop(gid, -1)
        return marked_pid

    def update_marked_points( self, page_selection: np.ndarray, gids: np.ndarray, cid: int ):
        add_pids, remove_ids = [], []
        for id in range( gids.shape[0] ):
            if page_selection[id]:
                add_pids.append(   self.mark_point( gids[id], cid ) )
            else:
                rpid = self.unmark_point( gids[id] )
                if rpid >= 0: remove_ids.append( rpid )
        if len( add_pids ):
            lm().addMarker( Marker( "marker", add_pids, cid ) )
        lm().deletePids( remove_ids )
        self.update_marker_plot()

    @log_timing
    def addMarker(self, marker: Marker ):
        if self.initialized:
            self.clear_transients()
            lgm().log(f" *** PointCloudManager-> ADD MARKER[{marker.size}], cid = {marker.cid}, #pids={marker.gids.size}")
            for gid in marker.gids:
                self.mark_point( gid, marker.cid )
            self.update_marker_plot()

    def deleteMarkers( self, gids: List[int], **kwargs ):
        if self.initialized:
            plot = kwargs.get('plot',False)
            for gid in gids:
                self.marker_gids.pop(gid, 0)
            if plot: self.update_marker_plot()

    def update_marker_plot(self):
        if self.marker_points is None:
            self.marker_points = p3js.Points( geometry=self.getMarkerGeometry(), material=self.marker_material )
            self.scene.add( [self.marker_points] )
            ipw.jslink((self.scene_controls['marker.material.size'],   'value'),  (self.marker_points.material, 'size') )
            ipw.jslink((self.scene_controls['marker.material.opacity'], 'value'), (self.marker_points.material, 'opacity') )
            self.marker_picker = p3js.Picker(controlling=self.marker_points, event='click')
            self.marker_picker.observe( partial( self.on_pick, True ), names=['point'] )
            self.renderer.controls = self.renderer.controls + [self.marker_picker]
        else:
            self.marker_points.geometry = self.getMarkerGeometry()

        if self.probe_points is None:
            self.probe_points = p3js.Points( geometry=self.getMarkerGeometry( probes=True ), material=self.probe_material )
            self.scene.add( [self.probe_points] )
            ipw.jslink((self.scene_controls['probe.material.size'],   'value'),  (self.probe_points.material, 'size') )
            ipw.jslink((self.scene_controls['probe.material.opacity'], 'value'), (self.probe_points.material, 'opacity') )
        else:
            self.probe_points.geometry = self.getMarkerGeometry( probes=True )

    @property
    def xyz(self)-> xa.DataArray:
        if self._xyz is None: self.init_data()
        return self._xyz

    def xyz(self, data_array: Union[xa.DataArray,np.ndarray] ):
        self._xyz = self.xyz.copy( data=data_array ) if (type(data_array) == np.ndarray) else data_array
        self._bounds = []
        self.voxelizer = Voxelizer( self._xyz, 0.1*self.scale )
        self.point_locator = self.xyz.values.sum(axis=1)

    def initialize_points(self):
        self._xyz: xa.DataArray = self.empty_pointset

    @property
    def empty_pointset(self) -> xa.DataArray:
        return xa.DataArray( np.empty(shape=[0, 3], dtype=np.float32), dims=["samples","model"] )

    @property
    def empty_pids(self) -> np.ndarray:
        return np.empty(shape=[0], dtype=np.int32)

    @exception_handled
    def init_data(self, **kwargs):
        from spectraclass.reduction.embedding import ReductionManager, rm
        from spectraclass.data.base import dm
        model_data: Optional[xa.DataArray] = dm().getModelData()

        if (model_data is not None) and (model_data.shape[0] > 1):
            lgm().log(f"UMAP.init: model_data{model_data.dims} shape = {model_data.shape}",print=True)
            # flow: ActivationFlow = afm().getActivationFlow()
            # if flow is None: return False
            # node_data = model_data if refresh else None
            # flow.setNodeData( node_data )
            embedding = rm().umap_init( model_data, **kwargs )
            self.xyz = self.pnorm(embedding)
        else:
            lgm().log(f"UMAP.init: model_data is empty",print=True)
            ecoords = dict( samples=[], model=np.arange(0,3) )
            attrs = {} if (model_data is None) else model_data.attrs
            self.xyz = xa.DataArray( np.empty([0,3]), dims=['samples','model'], coords=ecoords, attrs=attrs )

    def pnorm(self, point_data: xa.DataArray) -> xa.DataArray:
        return (point_data - point_data.mean()) * (self.scale / point_data.std())

    # def pnorm(self, data: xa.DataArray, dim: int = 1) -> xa.DataArray:
    #     dave, dmag = np.nanmean(data.values, keepdims=True, axis=dim), np.nanstd(data.values, keepdims=True, axis=dim)
    #     normed_data = (data.values - dave) / dmag
    #     return data.copy(data=normed_data)

    def getColors( self, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.gui.spatial.map import MapManager, mm
        cdata: Optional[xa.DataArray] = kwargs.get( 'cdata', mm().getPointData( current_frame=True ) )
        if cdata.ndim > 1:
            cdata, _, _ = tm().getBlock().raster2points(cdata)
        vr = mm().get_color_bounds( cdata )
        norm = Normalize( vr['vmin'], vr['vmax'] )
        tmask: np.ndarray = mm().block.get_threshold_mask(raster=False)
        if (tmask is not None) and (tmask.shape[0] == cdata.shape[0]):
            cdata = cdata[ tmask ]
        lgm().log( f"getColors: norm cdata shape = {cdata.shape}, dims={cdata.dims}, crange={[norm.vmin,norm.vmax]}")
        mapper = plt.cm.ScalarMappable( norm = norm, cmap="jet" )
        colors = mapper.to_rgba( cdata.values )[:, :-1] * 255
        if self.pick_point >= 0:
            pid = self.voxelizer.gid2pid( self.pick_point )
            colors[ pid ] = [255] * colors.shape[1]
        return colors.astype(np.uint8)

    def getColors1( self, **kwargs ):
        from spectraclass.gui.spatial.map import MapManager, mm
        norm: Normalize = kwargs.get('norm')
        cdata: Optional[xa.DataArray] = mm().getPointData( current_frame=True )
        if norm is None:
            vr = mm().get_color_bounds( cdata )
            norm = Normalize( vr['vmin'], vr['vmax'] )
        tmask: np.ndarray = mm().block.get_threshold_mask(raster=False)
        if (tmask is not None) and (tmask.shape[0] == cdata.shape[0]):
            cdata = cdata[ tmask ]
        lgm().log( f"getColors: norm cdata shape = {cdata.shape}, dims={cdata.dims}, crange={[norm.vmin,norm.vmax]}")
        mapper = plt.cm.ScalarMappable( norm = norm, cmap="jet" )
        colors = mapper.to_rgba( cdata.values )[:, :-1] * 255
        if self.pick_point >= 0:
            pid = self.voxelizer.gid2pid( self.pick_point )
            colors[ pid ] = [255] * colors.shape[1]
        return colors.astype(np.uint8)

    @exception_handled
    def getGeometry( self, **kwargs ) -> Optional[p3js.BufferGeometry]:
        geometry_data = kwargs.get( 'init_data', None )
        if geometry_data is None: geometry_data = self.xyz
        colors = self.getColors(**kwargs)
        lgm().log(f" *** getGeometry: xyz shape = {geometry_data.shape}, color shape = {colors.shape}")
        attrs = dict( position = p3js.BufferAttribute( geometry_data, normalized=False ),
                      color =    p3js.BufferAttribute( list(map(tuple, colors))) )
        return p3js.BufferGeometry( attributes=attrs )

    @exception_handled
    def getMarkerGeometry( self, **kwargs ) -> p3js.BufferGeometry:
        probes = kwargs.get('probes',False)
        gids = np.array(list(self.probe_gids.keys() if probes else self.marker_gids.keys()))
        if gids.size == 0:
            positions = np.empty( shape=[0,3], dtype=np.int32 )
            colors = np.empty( shape=[0,3], dtype=np.uint8 )
        else:
            srange, ssize = [self.xyz.samples.values.min(),self.xyz.samples.values.max()], self.xyz.samples.values.size
            xrange = [ self.xyz.values.min(), self.xyz.values.max() ]
            lgm().log(f"*** getMarkerGeometry->gids: size = {gids.size}, gid range = {[gids.min(),gids.max()]}; samples: size = {ssize}, range={srange}; data range = {xrange}")
            mask = np.isin( self.xyz.samples.values, gids, assume_unique=True )
            sgids = self.xyz.samples.values[mask]
            cids = self.probe_gids if probes else self.marker_gids
            colors = lm().get_rgb_colors( [cids[gid] for gid in sgids], probes)
            positions = self.xyz.values[mask]
        attrs = dict( position=p3js.BufferAttribute( positions, normalized=False ), color=p3js.BufferAttribute( colors ) )
        return p3js.BufferGeometry( attributes=attrs )

    def createPoints( self, **kwargs ):
        points_geometry = self.getGeometry( **kwargs )
        assert not points_geometry is None, "Initialization error, please check log file for details"
        points_material = p3js.PointsMaterial( vertexColors='VertexColors', transparent=True )
        self.points = p3js.Points( geometry=points_geometry, material=points_material )

    def getControlsWidget(self) -> ipw.DOMWidget:
        self.scene_controls['point.material.size']     = ipw.FloatSlider( value=0.015 * self.scale, min=0.0, max=0.05 * self.scale, step=0.0002 * self.scale)
        self.scene_controls['point.material.opacity']  = ipw.FloatSlider( value=1.0, min=0.0, max=1.0, step=0.01 )
        self.scene_controls['marker.material.size']    = ipw.FloatSlider( value=0.05 * self.scale, min=0.0, max=0.1 * self.scale, step=0.001 * self.scale )
        self.scene_controls['marker.material.opacity'] = ipw.FloatSlider( value=1.0, min=0.0, max=1.0, step=0.01 )
        self.scene_controls['probe.material.size']    = ipw.FloatSlider( value=0.05 * self.scale, min=0.0, max=0.2 * self.scale, step=0.001 * self.scale )
        self.scene_controls['probe.material.opacity'] = ipw.FloatSlider( value=1.0, min=0.0, max=1.0, step=0.01 )
        self.scene_controls['window.scene.background'] = ipw.ColorPicker( value="black" )
        ipw.jslink( (self.scene_controls['point.material.size'],     'value'),   ( self.points.material, 'size' ) )
        ipw.jslink( (self.scene_controls['point.material.opacity'],  'value'),   ( self.points.material, 'opacity' ) )
        ipw.jslink( (self.scene_controls['window.scene.background'], 'value'),   ( self.scene, 'background') )
        return ipw.VBox( [ ipw.HBox( [ self.control_label(name), ctrl ] ) for name, ctrl in self.scene_controls.items() ] )

    def control_label(self, name: str ) -> ipw.Label:
        toks = name.split(".")
        return ipw.Label( f"{toks[0]} {toks[2]}" )

    def _get_gui( self ) -> ipw.DOMWidget:
        ecoords = dict(samples=[], model=np.arange(0, 3))
        init_data = xa.DataArray(np.empty([0, 3]), dims=['samples', 'model'], coords=ecoords, attrs={})
        self.createPoints( init_data=init_data )
        self.scene = p3js.Scene( children=[ self.points, self.camera, p3js.AmbientLight(intensity=0.8)  ] )
        self.renderer = p3js.Renderer( scene=self.scene, camera=self.camera, controls=[self.orbit_controls], width=800, height=500 )
        self.point_picker = p3js.Picker(controlling=self.points, event='click')
        self.point_picker.observe( partial( self.on_pick, False ), names=['point'])
        self.renderer.controls = self.renderer.controls + [self.point_picker]
        self.control_panel = self.getControlsWidget()
        return ipw.VBox( [ self.renderer, self.control_panel ]  )

    @exception_handled
    def on_pick(self, highlight: bool, event: Dict ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.gui.unstructured.table import tbm
        point = tuple( event["new"] )
        ppid = self.voxelizer.get_gid(point)
        if ppid >= 0:
            self.pick_point = ppid
            if mm().initialized():
                if highlight:   pos = mm().highlight_points( [self.pick_point], [0] )
                else:           pos = mm().mark_point( self.pick_point, cid=0 )
                lgm().log( f"  on_pick[highlight={highlight}] pick_point={self.pick_point} pos={pos}")
            if tbm().active:
                tbm().mark_point( self.pick_point, 0, point )
            self.mark_point( self.pick_point, 0 )
            self.points.geometry = self.getGeometry()

    def gui(self, **kwargs ) -> ipw.DOMWidget:
        if self._gui is None:
            self._gui = self._get_gui()
        return self._gui

    def get_index_from_point(self, point: List[float] ) -> int:
        spt = sum(point)
        loc_array: np.ndarray = np.abs( self.point_locator - spt )
        indx = np.argmin( loc_array )[0]
        lgm().log( f"get_index_from_point[{indx}]: Loc array range=[{loc_array.min()},{loc_array.max()}], spt={loc_array[indx]}, pos={self.xyz[indx]}")
        return indx

    @property
    def initialized(self):
        return self._xyz is not None

    @exception_handled
    def update_plot(self, **kwargs) -> bool:
        t0 = time.time()
        if 'points' in kwargs:
            embedding = kwargs['points']
            lgm().log( f"PCM->plot embedding: shape = {embedding.shape}")
            self.xyz = self.pnorm( embedding )
        if not self.initialized:
            return False
        t1 = time.time()
        if self._gui is not None:
            geometry =  self.getGeometry( **kwargs )
            t2 = time.time()
            if geometry is not None:
                self.points.geometry = geometry
                if self.marker_points is not None:
                    self.marker_points.geometry = self.getMarkerGeometry()
                if self.probe_points is not None:
                    self.probe_points.geometry = self.getMarkerGeometry(probes=True)
            t3 = time.time()
            lgm().log( f" *** update point cloud data: time = {t3-t2} {t2-t1} {t1-t0} " )
            return True

    def clear(self):
        if self.initialized:
            lgm().log(f"  $CLEAR: PCM")
            self.marker_gids = {}
            self.probe_gids = {}
            if self.marker_points is not None:
                self.marker_points.geometry = self.getMarkerGeometry()
            if self.probe_points is not None:
                self.marker_points.geometry = self.getMarkerGeometry(probes=True)


    def color_by_value(self, values: np.ndarray = None, **kwargs):
        if self.initialized:
            if values is not None:
                lgm().log(f" $$$color_by_value: data shape = {values.shape} ")
                self._color_values = ma.masked_invalid(values)
            if self._color_values is not None:
                colors = self._color_values.filled(-1)
                lgm().log( f" $$$color_by_value (shape:{colors.shape}) ")
                self.update_plot( cdata=colors, **kwargs)

    def get_color_bounds(self):
        from spectraclass.data.spatial.manager import SpatialDataManager
        (ave, std) = (self._color_values.mean(), self._color_values.std())
        return (ave - std * SpatialDataManager.colorstretch, ave + std * SpatialDataManager.colorstretch, ave, std)

    def reset(self):
        self._xyz = None

    def refresh(self):
        if self._gui is not None:
            if self.init_data( refresh=True ):
                self.update_plot()


