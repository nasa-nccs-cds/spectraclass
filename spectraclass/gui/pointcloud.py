import pythreejs as p3js
from functools import partial
from panel.layout import Panel
from panel.widgets import FloatSlider
import ipywidgets as ipw
import panel as pn
import holoviews as hv
import numpy as np
from spectraclass.reduction.embedding import ReductionManager, rm
import time, math, os, sys
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.gui.spatial.widgets.markers import Marker
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
import numpy.ma as ma
import pickle, xarray as xa
from spectraclass.data.spatial.voxels import Voxelizer
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import traitlets as tl
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.model.base import SCSingletonConfigurable

def pcm() -> "PointCloudManager":
    return PointCloudManager.instance()

def asarray( data: Union[np.ndarray,Iterable], dtype  ) -> np.ndarray:
    if isinstance( data, np.ndarray ): return data
    else: return np.array( list(data), dtype = np.dtype(dtype) )

def autocorr( data: xa.DataArray, dims: Tuple[int,int]=(0,1) ) -> float:
    from scipy.stats import pearsonr
    dslices = [ data.values[:,iD] for iD in dims ]
    acorr, _ = pearsonr(*dslices)
    return acorr

class PointCloudManager(SCSingletonConfigurable):

    color_map = tl.Unicode("gist_rainbow").tag(config=True)  # "gist_rainbow" "jet"
#    opacity = tl.Float( 'opacity', min=0.0, max=1.0 ).tag(sync=True)

    def __init__( self):
        super(PointCloudManager, self).__init__()
        self._gui = None
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
        self.control_panel: Panel = None
        self.size_control: FloatSlider = None
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
        self.colorstretch = 2.0
        self.block_watcher = None

    @exception_handled
    def set_block_callback(self,*events):
        for event in events:
            if (event.name == 'index') and (event.new >= 0):
                self.update_plot( bindex=tm().bi2c(event.new) )

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
            self.scene_controls[ 'marker.material.size'   ].jslink( target=self.marker_points.material,  value="size" )
            self.scene_controls[ 'marker.material.opacity'].jslink( target=self.marker_points.material,  value="opacity" )
            self.marker_picker = p3js.Picker(controlling=self.marker_points, event='click')
            self.marker_picker.observe( partial( self.on_pick, True ), names=['point'] )
            self.renderer.controls = self.renderer.controls + [self.marker_picker]
        else:
            self.marker_points.geometry = self.getMarkerGeometry()

        if self.probe_points is None:
            self.probe_points = p3js.Points( geometry=self.getMarkerGeometry( probes=True ), material=self.probe_material )
            self.scene.add( [self.probe_points] )
            self.scene_controls[ 'probe.material.size'   ].jslink( target=self.probe_points.material,   value = 'size' )
            self.scene_controls[ 'probe.material.opacity'].jslink( target=self.probe_points.material,   value = 'opacity' )
        else:
            self.probe_points.geometry = self.getMarkerGeometry( probes=True )

    @property
    def xyz(self)-> xa.DataArray:
        if self._xyz is None: self.init_data()
        return self._xyz

    @property
    def frame_size(self):
        return self._xyz.values.max()

    @xyz.setter
    def xyz(self, data_array: Union[xa.DataArray,np.ndarray] ):
        self._xyz = self._xyz.copy( data=data_array ) if (type(data_array) == np.ndarray) else data_array
        self._bounds = []
        self.voxelizer = Voxelizer( self._xyz, 0.1*self.scale )
        self.point_locator = self.xyz.values.sum(axis=1)
        lgm().log( f"PCM: set points data, shape = {self._xyz.shape}")

    def initialize_points(self):
        self._xyz: xa.DataArray = self.empty_pointset

    @property
    def empty_pointset(self) -> xa.DataArray:
        return xa.DataArray( np.empty(shape=[0, 3], dtype=np.float32), dims=["samples","model"] )

    def toxa(self, data: np.ndarray ) -> xa.DataArray:
        return xa.DataArray( data, dims=["samples","model"] )

    @property
    def empty_pids(self) -> np.ndarray:
        return np.empty(shape=[0], dtype=np.int32)

    @exception_handled
    def init_data( self, **kwargs ):
        from spectraclass.reduction.embedding import ReductionManager, rm
        from spectraclass.data.base import dm
        model_data: Optional[xa.DataArray] = dm().getModelData()

        if (model_data is not None) and (model_data.shape[0] > 1):
            use_umap = model_data.shape[1] > 3
            lgm().log(f"PCM: model_data{model_data.dims} shape = {model_data.shape}, use_umap={use_umap}")
            # flow: ActivationFlow = afm().getActivationFlow()
            # if flow is None: return False
            # node_data = model_data if refresh else None
            # flow.setNodeData( node_data )
            embedding = rm().umap_init( model_data, **kwargs ) if use_umap else model_data
            self.xyz = self.pnorm(embedding)
            lgm().log( f"PCM: autocorr = {autocorr(self.xyz,(0,1)):.2f} {autocorr(self.xyz,(0,2)):.2f} {autocorr(self.xyz,(1,2)):.2f}")
        else:
            lgm().log(f"UMAP.init: model_data is empty",print=True)
            ecoords = dict( samples=[], model=np.arange(0,3) )
            attrs = {} if (model_data is None) else model_data.attrs
            self.xyz = xa.DataArray( np.empty([0,3]), dims=['samples','model'], coords=ecoords, attrs=attrs )

#        self.block_watcher = tm().block_selection.param.watch(self.set_block_callback, ['index'], onlychanged=True)

    def pnorm(self, point_data: xa.DataArray) -> xa.DataArray:
        return (point_data - point_data.mean()) * (self.scale / point_data.std())

    # def pnorm(self, data: xa.DataArray, dim: int = 1) -> xa.DataArray:
    #     dave, dmag = np.nanmean(data.values, keepdims=True, axis=dim), np.nanstd(data.values, keepdims=True, axis=dim)
    #     normed_data = (data.values - dave) / dmag
    #     return data.copy(data=normed_data)

    def get_color_bounds( self, raster: xa.DataArray ):
        ave = np.nanmean( raster.values )
        std = np.nanstd(  raster.values )
        nan_mask = np.isnan( raster.values )
        nnan = np.count_nonzero( nan_mask )
        lgm().log( f" **get_color_bounds: mean={ave}, std={std}, #nan={nnan}" )
        return dict( vmin= ave - std * self.colorstretch, vmax= ave + std * self.colorstretch  )

    def getColors( self, **kwargs ):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.gui.spatial.map import MapManager, mm
        cdata: Optional[xa.DataArray] = kwargs.get( 'cdata', mm().getPointData( current_frame=True ) )
        if cdata.ndim > 1:
            cdata, _, _ = tm().getBlock().raster2points(cdata)
        vr = self.get_color_bounds( cdata )
        norm = Normalize( vr['vmin'], vr['vmax'] )
        # tmask: np.ndarray = mm().block.get_threshold_mask(raster=False)
        # if (tmask is not None) and (tmask.shape[0] == cdata.shape[0]):
        #     cdata = cdata[ tmask ]
        lgm().log( f"getColors: norm cdata shape = {cdata.shape}, dims={cdata.dims}" ) # , crange={[norm.vmin,norm.vmax]}")
        mapper = plt.cm.ScalarMappable(  norm = norm, cmap="jet" )
        colors = mapper.to_rgba( cdata.values )[:, :-1] * 255
        if self.pick_point >= 0:
            pid = self.voxelizer.gid2pid( self.pick_point )
            colors[ pid ] = [255] * colors.shape[1]
        return colors.astype(np.uint8)

    @exception_handled
    def getGeometry( self, **kwargs ) -> Optional[p3js.BufferGeometry]:
        geometry_data = kwargs.get( 'init_data', self.xyz )
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

    def getControlsWidget(self) -> Panel:
        self.scene_controls['point.material.size']     = ipw.FloatSlider( description="point size",     value=0.015 * self.scale,  min=0.0, max=0.05 * self.scale, step=0.0002 * self.scale)
        self.scene_controls['point.material.opacity']  = ipw.FloatSlider( description="point opacity",  value=1.0,                 min=0.0, max=1.0,               step=0.01 )
        self.scene_controls['marker.material.size']    = ipw.FloatSlider( description="marker size",    value=0.05 * self.scale,   min=0.0, max=0.1 * self.scale,  step=0.001 * self.scale )
        self.scene_controls['marker.material.opacity'] = ipw.FloatSlider( description="marker opacity", value=1.0,                 min=0.0, max=1.0,               step=0.01 )
  #      self.scene_controls['probe.material.size']     = ipw.FloatSlider( description="probe size",     value=0.05 * self.scale,   min=0.0, max=0.2 * self.scale,  step=0.001 * self.scale )
  #      self.scene_controls['probe.material.opacity']  = ipw.FloatSlider( description="probe opacity",  value=1.0,                 min=0.0, max=1.0,               step=0.01 )
        self.scene_controls['window.scene.background'] = ipw.ColorPicker() # pn.widgets.ColorPicker( value="black" )
        # self.scene_controls[ 'point.material.size'   ].observe(  partial( self.update_parameter, 'point.size',    names=['value'] )  )
        # self.scene_controls[ 'point.material.opacity'].observe(  partial( self.update_parameter, 'point.opacity', names=['value'] ) )
        # self.scene_controls[ 'window.scene.background'].observe( partial( self.update_parameter, 'background',    names=['value'] ) )
        self.link_controls()
        controls = ipw.VBox( list(self.scene_controls.values()) )
        return pn.Column( controls )

    def link_controls(self):
        for name, ctrl in self.scene_controls.items():
            toks = name.split(".")
            if toks[1] == "scene":
                object = self.scene
            elif toks[1] == "material":
                if toks[0] == "point":
                    object = self.points.material
                else:
                    object = self.marker_points.material if (self.marker_points is not None) else None
            else:
                raise Exception( f"Unrecognized control domain: {toks[1]}")
            if object is not None:
                ipw.jslink( (ctrl, 'value'), (object, toks[2]) )

    # def update_parameter(self, name, event ):
    #     print( f"update_parameter[{name}] = {event}")
    #     if name == 'point.opacity': self.points.material.opacity = event['new']


        # self.scene_controls[ 'point.material.size'   ].observe( target=self.points.material, value="size"  )
        # self.scene_controls[ 'point.material.opacity'].observe( target=self.points.material, value="opacity" )
        # self.scene_controls[ 'window.scene.background'].observe( target=self.scene, value="background" )

    # def update_parameter(self, name: str, value: float ):
    #     param = self.get_parametere( name )
    #     param = value
    #
    # def update_parameter(self, name ):
    #     toks = name.split(".")
    #     if toks[0] == 'point': return

    def get_frame(self) -> p3js.Mesh:
        size = self.frame_size
        box = p3js.BoxLineGeometry(size,size,size)
        material = p3js.MeshBasicMaterial( dict(color= 0x555555) )
        return p3js.Mesh( box, material)

    @exception_handled
    def _get_gui( self ) -> Panel:
        self.init_data(refresh=True)
        self.createPoints()
        self.frame = self.get_frame()
        self.scene = p3js.Scene( children=[ self.points, self.frame, self.camera, p3js.AmbientLight(intensity=0.8)  ] )
        self.renderer = p3js.Renderer( scene=self.scene, camera=self.camera, controls=[self.orbit_controls], width=800, height=500 )
        self.point_picker = p3js.Picker(controlling=self.points, event='click')
        self.point_picker.observe( partial( self.on_pick, False ), names=['point'])
        self.renderer.controls = self.renderer.controls + [self.point_picker]
        self.control_panel = self.getControlsWidget()
        return pn.Column( self.renderer, self.control_panel  )

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

    def gui(self, **kwargs ) -> Panel:
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
        from spectraclass.data.spatial.tile.tile import Block
        t0 = time.time()
        if 'points' in kwargs:
            embedding = kwargs['points']
            lgm().log( f"PCM->plot embedding: shape = {embedding.shape}")
            self.xyz = self.pnorm( embedding )
        elif 'bindex' in kwargs:
            block: Block = tm().getBlock(kwargs['bindex'])
            model_data = block.getModelData( raster=False )
            embedding = rm().umap_init( model_data, **kwargs ) if model_data.shape[1] > 3 else model_data
            self.xyz = self.pnorm(embedding)
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

    # def get_color_bounds(self):
    #     from spectraclass.data.spatial.manager import SpatialDataManager
    #     (ave, std) = (self._color_values.mean(), self._color_values.std())
    #     return (ave - std * SpatialDataManager.colorstretch, ave + std * SpatialDataManager.colorstretch, ave, std)

    def reset(self):
        self._xyz = None

    def refresh(self):
        if self._gui is not None:
            if self.init_data( refresh=True ):
                self.update_plot()


