import ipywidgets as ipw
import pythreejs as p3js
import matplotlib.pyplot as plt
import time, math, os, sys, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from matplotlib import cm
import xarray as xa
from matplotlib import colors
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import traitlets as tl
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.model.base import SCSingletonConfigurable

def pcm() -> "PointCloudManager":
    return PointCloudManager.instance()

class PointCloudManager(SCSingletonConfigurable):

    color_map = tl.Unicode("gist_rainbow").tag(config=True)  # "gist_rainbow" "jet"

    def __init__( self):
        super(PointCloudManager, self).__init__()
        self._gui = None
        self.xyz: np.ndarray = None
        self.points = None
        self.scene = None
        self.renderer = None
        self.widgets = None
        self._n_point_bins = 27
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
        self._binned_points: List[np.ndarray] = [self.empty_pointset for ic in range(self._n_point_bins)]
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

    def get_bin_colors(self, cmname: str, invert=False):
        x: np.ndarray = np.linspace(0.0, 1.0, self._n_point_bins)
        cmap = cm.get_cmap(cmname)(x).tolist()
        return cmap[::-1] if invert else cmap

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

    def getColors(self, cmap=None, colors=None ):
        if colors is None:
            colors = np.repeat([[255, 125, 0]], self.xyz.shape[0], axis=0)
        else:
            s_m = plt.cm.ScalarMappable(cmap=cmap)
            colors = s_m.to_rgba(colors)[:, :-1] * 255
        return colors.astype(np.uint8)

    def getPoints(self, colors):
        attrs = dict( position=p3js.BufferAttribute( self.xyz, normalized=False ),
                      color=p3js.BufferAttribute(list(map(tuple, colors))))
        points_geometry = p3js.BufferGeometry( attributes=attrs )
        points_material = p3js.PointsMaterial( vertexColors='VertexColors')
        return p3js.Points( geometry=points_geometry, material=points_material)

    def getControlsWidget(self):
        initial_point_size = self.xyz.ptp() / 100
        size = ipw.FloatSlider( value=initial_point_size, min=0.0, max=initial_point_size * 10, step=initial_point_size / 100)
        ipw.jslink( (size,'value'), (self.points.material,'size') )
        color = ipw.ColorPicker( value="black" )
        ipw.jslink( (color,'value'), (self.scene,'background') )
        psw = ipw.HBox( [ ipw.Label('Point size:'), size ] )
        bcw = ipw.HBox( [ ipw.Label('Background color:'), color ] )
        return ipw.VBox( [ psw, bcw ] )

    def _get_gui( self, **kwargs ) -> ipw.DOMWidget:
        colors = self.getColors()
        self.points = self.getPoints( colors )
        self.scene = p3js.Scene( children=[ self.points, self.camera ] )
        self.renderer = p3js.Renderer( scene=self.scene, camera=self.camera, controls=[self.orbit_controls], width=1000, height=600 )
        self.widgets = self.getControlsWidget()
        return ipw.VBox( [ self.renderer, self.widgets ]  )

    def gui(self, **kwargs ) -> ipw.DOMWidget:
        if self._gui is None:
            self.init_data( **kwargs )
            self._gui = self._get_gui( **kwargs  )
        return self._gui
