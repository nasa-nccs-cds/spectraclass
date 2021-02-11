import time, math, os, sys, numpy as np
from spectraclass.reduction.embedding import ReductionManager
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from matplotlib import cm
from itkwidgets import view
from itkwidgets.widget_viewer import Viewer
import xarray as xa
from matplotlib import colors
import numpy.ma as ma
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm, exception_handled
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable, Marker
from spectraclass.model.labels import LabelsManager, lm

def pcm() -> "PointCloudManager":
    return PointCloudManager.instance()

def asarray( data: Union[np.ndarray,Iterable], dtype  ) -> np.ndarray:
    if isinstance( data, np.ndarray ): return data
    else: return np.array( list(data), dtype = np.dtype(dtype) )

class PointCloudManager(SCSingletonConfigurable):

    color_map = tl.Unicode("gist_rainbow").tag(config=True)  # "gist_rainbow" "jet"

    def __init__(self):
        super(PointCloudManager, self).__init__()
        self._gui: Viewer = None
        self._n_point_bins = 27
        self._color_values = None
        self.reduced_opacity = 0.111111
        self.standard_opacity = 0.411111
        self.initialize_points()

    def initialize_points(self):
        self._binned_points: List[np.ndarray] = [self.empty_pointset for ic in range(self._n_point_bins)]
        self._points: np.ndarray = self.empty_pointset
        self._marker_points: List[np.ndarray] = None
        self._marker_pids: List[np.ndarray] = None
        self._color_values = None

    def initialize_markers(self, reset= False ):
        if (self._marker_points is None) or reset:
            nLabels = lm().nLabels
            self._marker_points: List[np.ndarray] = [ self.empty_pointset for ic in range( nLabels ) ]
            self._marker_pids: List[np.ndarray] = [ self.empty_pids for ic in range( nLabels ) ]
            lgm().log( f"PCM.initialize_markers, sizes = {[m.size for m in self._marker_points]}")

    @property
    def empty_pointset(self) -> np.ndarray:
        return np.empty(shape=[0, 3], dtype=np.float)

    def get_bin_colors( self, cmname: str, invert = False ):
        x: np.ndarray = np.linspace( 0.0, 1.0, self._n_point_bins )
        cmap = cm.get_cmap(cmname)(x).tolist()
        return cmap[::-1] if invert else cmap

    @property
    def empty_pids(self) -> np.ndarray:
        return np.empty(shape=[0], dtype=np.int)

    def init_data( self, **kwargs  ):
        from spectraclass.data.base import DataManager
        project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("points")
        reduced_data: xa.DataArray = project_dataset.reduction
        reduced_data.attrs['dsid'] = project_dataset.attrs['dsid']
        lgm().log( f"UMAP init, init data shape = {reduced_data.shape}")
        embedding = ReductionManager.instance().umap_init( reduced_data, **kwargs  )
        self._points = self.normalize( embedding )
        self.initialize_markers()

    def normalize(self, point_data: np.ndarray ):
        return ( point_data - point_data.mean() ) / point_data.std()

    def reembed(self, embedding ):
        self.clear_bins()
        self.update_plot( points = embedding )

    def update_plot( self, **kwargs ):
        if 'points' in kwargs: self._points = self.normalize( kwargs['points'] )
        if self._gui is not None:
            lgm().log(f"Updating point sets, sizes: {[ps.shape[0] for ps in self.point_sets]}")
            self._gui.point_sets = self.point_sets
            self._gui.update_rendered_image()

    def on_selection(self, selection_event: Dict ):
        selection = selection_event['pids']
        self.update_markers(selection)
        self.update_plot()

    def update_points(self, new_points: np.ndarray ):
        self.update_markers(points=new_points)
        self.color_by_value()

    def update_markers(self, pids: List[int] = None, **kwargs ):
        if pids is None:
            if 'points' in kwargs: self._points = self.normalize( kwargs['points'] )
            self.initialize_markers(True)
            for marker in lm().markers:
                self._marker_pids[ marker.cid ] = np.append( self._marker_pids[ marker.cid ], marker.pids, 0 )
                self._marker_points[ marker.cid ] = self._points[ self._marker_pids[ marker.cid ], : ]
        else:
            self._marker_pids[0] = np.array(pids)
            self._marker_points[0] = self._points[ self._marker_pids[ 0 ], : ]
            lgm().log( f"  ***** POINTS- mark_points[0], #pids = {len(pids)}")

    @exception_handled
    def update_marked_points(self, cid: int = -1, **kwargs ):
        from spectraclass.gui.control import UserFeedbackManager, ufm
        if self._points is None:
            ufm().show( "Can't mark points in PointCloudManager which is not initialized", "red")
        else:
            icid = cid if cid >= 0 else lm().current_cid
            self.initialize_markers()
            self.clear_points(0)
            self._marker_pids[icid] = asarray( kwargs.get( 'pids', lm().getPids(icid) ), np.int )
            self._marker_points[ icid ] = self._points[ self._marker_pids[icid], : ]
            lgm().log(f"PointCloudManager.add_marked_points: cid = {icid}, #marked-points[cid]: [{self._marker_pids[icid].size}]")
        self.set_base_points_alpha( self.reduced_opacity )
        self.update_plot()

    def clear_bins(self):
        lgm().log(f"PointCloudManager.clear_bins")
        for iC in range( 0, self._n_point_bins ):
            self._binned_points[iC] = self.empty_pointset
        if self._gui.point_set_opacities[0] == self.reduced_opacity:
            self.set_base_points_alpha( self.standard_opacity )
        self.update_plot()

    def clear(self):
        self.clear_bins()
        self.initialize_markers( True )
        self.update_plot()

    def set_bin_colors(self, bin_colors: List[str] ):
        from matplotlib import colors
        new_colors = self.standard_colors.copy()
        for iC, color in enumerate(bin_colors):
            new_colors[iC+1] = colors.to_rgb( color )
        self._gui.point_set_colors = new_colors

    def color_by_index(self, indices: np.ndarray, colors: List, **kwargs):
        pts: np.ndarray = ma.masked_invalid(self._points).filled(-1)
        imax = indices.max()
        for iC in range( 0, self._n_point_bins ):
            self._binned_points[iC] = pts[ (indices == iC) ] if iC <= imax else self.empty_pointset
        self.set_base_points_alpha(self.reduced_opacity)
        self.set_bin_colors( colors )
        self.update_plot(**kwargs)

    def color_by_value( self, values: np.ndarray = None, **kwargs ):
        is_distance = kwargs.get( 'distance', False )
        if values is not None:
            lgm().log(f" $$$color_by_value[distance:{is_distance}]: data shape = {values.shape} ")
            self._color_values = ma.masked_invalid(values)
        if self._color_values is not None:
            colors = self._color_values.filled( -1 )
            (vmin, vmax), npb = ( ( 0.0, self._color_values.max() ) if is_distance else self.get_color_bounds() ), self._n_point_bins
            lgm().log( f" $$$color_by_value(shape:{colors.shape}) (vmin, vmax, npb) = {(vmin, vmax, npb)}, points (max, min, shape) = { (self._points.max(), self._points.min(), self._points.shape) }" )
            pts: np.ndarray = ma.masked_invalid( self._points ).filled( -1 )
            lspace: np.ndarray = np.linspace( vmin, vmax, npb+1 )
            for iC in range( 0, npb ):
                if iC == 0:          mask = colors <= lspace[iC+1]
                elif (iC == npb-1):  mask = ( colors > lspace[iC] ) & ( colors < sys.float_info.max )
                else:                mask = ( colors > lspace[iC] ) & ( colors <= lspace[iC+1] )
                self._binned_points[iC] = pts[ mask ]
#                lgm().log(f" $$$COLOR: BIN-{iC}, [ {lspace[iC]} -> {lspace[iC+1]} ], nvals = {self._binned_points[iC].shape[0]}, #mask-points = {np.count_nonzero(mask)}" )
            self.set_base_points_alpha( self.reduced_opacity )
            self._gui.point_set_colors = self.standard_colors
            self.update_plot(**kwargs)

    def get_color_bounds( self ):
        from spectraclass.data.spatial.manager import SpatialDataManager
        (ave, std)= ( self._color_values.mean(),  self._color_values.std() )
        return ( ave - std * SpatialDataManager.colorstretch, ave + std * SpatialDataManager.colorstretch  )

    @exception_handled
    def clear_pids(self, cid: int, pids: np.ndarray, **kwargs):
        lgm().log(f"PCM.clear_pids: marker_pids = {[ p.tolist() for p in self._marker_pids]}")
        if self._marker_pids is not None:
            dpts = np.vectorize(lambda x: x in pids)
            for iC, marker_pids in enumerate( self._marker_pids ):
                if (cid < 0) or (iC == cid):
                    if len( marker_pids ) > 0:
                        self._marker_pids[iC] = np.delete( self._marker_pids[iC], dpts(marker_pids) )
                        self._marker_points[iC] = self._points[self._marker_pids[iC], :] if len( self._marker_pids[iC] ) > 0 else self.empty_pointset
                        lgm().log(f"  --> cid={cid}, # pids cleared = {pids.size}, # remaining markers = {len( self._marker_pids[iC] )}")
                    else:
                        self._marker_points[iC] = self.empty_pointset

    def clear_points(self, icid: int, **kwargs ):
        if self._marker_pids is not None:
            pids = kwargs.get('pids', None )
            if pids is None:
                lgm().log(f"PCM.clear_points[{icid}]: empty_pointset")
                self._marker_points[icid] = self.empty_pointset
                self._marker_pids[icid] = self.empty_pids
            else:
                lgm().log(f"PCM.clear_points: cid={icid}, #pids={pids.size}")
                dpts = np.vectorize( lambda x: x in pids )
                dmask = dpts( self._marker_pids[icid] )
    #            lgm().log( f"clear_points.Mask: {self._marker_pids[icid]} -> {dmask}" )
                self._marker_pids[icid]  = np.delete( self._marker_pids[icid], dmask )
                self._marker_points[ icid ] = self._points[ self._marker_pids[icid], :] if len( self._marker_pids[icid] ) > 0 else self.empty_pointset
    #            lgm().log(f"clear_points: reduced marker_pids = {self._marker_pids[icid]} -> points = {self._marker_points[ icid ]}")

    @property
    def point_sets(self):
        self.initialize_markers()
        return [ self._points ] + self._binned_points + self._marker_points[::-1]

    def gui(self, **kwargs ):
        if self._gui is None:
            self.init_data()
            invert = False
            bin_colors = [ x[:3] for x in self.get_bin_colors( self.color_map, invert ) ]
            label_colors = [ colors.to_rgb(c) for c in lm().colors[::-1] ]
            self.standard_colors =   [ [1.0, 1.0, 1.0], ] + bin_colors + label_colors
            pt_alphas = [ self.standard_opacity ] * ( self._n_point_bins + 1 ) + [ 1.0 ] * lm().nLabels
            ptsizes = [1] + [1]*self._n_point_bins + [8]*lm().nLabels
            self._gui = view( point_sets = self.point_sets, point_set_sizes=ptsizes, point_set_colors=self.standard_colors, point_set_opacities=pt_alphas, background=[0,0,0] )
            self._gui.layout = dict( width= '100%', flex= '1 0 1200px' )
        return self._gui

    def set_base_points_alpha( self, alpha: float ):
        alphas = list( self._gui.point_set_opacities )
        alphas[0] = alpha
        self._gui.point_set_opacities = alphas
  #      lgm().log(f"Set point set opacities: {self._gui.point_set_opacities}")
        self.update_plot()

    def refresh(self):
        self.clear_bins()
        self.initialize_points()
        self.initialize_markers( True )
        self.init_data()
        self.update_plot()
