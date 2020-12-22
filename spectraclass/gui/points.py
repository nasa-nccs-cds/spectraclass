import time, math, numpy as np
from spectraclass.data.base import DataManager
from spectraclass.reduction.embedding import ReductionManager
from typing import List, Union, Tuple, Optional, Dict, Callable
from matplotlib import cm
from itkwidgets import view
from itkwidgets.widget_viewer import Viewer
import xarray as xa
import numpy.ma as ma
import traitlets.config as tlc
from spectraclass.model.base import SCConfigurable, Marker
from spectraclass.model.labels import LabelsManager

class PointCloudManager(tlc.SingletonConfigurable, SCConfigurable):

    def __init__(self):
        super(PointCloudManager, self).__init__()
        self._gui: Viewer = None
        self._n_point_bins = 27
        self.initialize_points()

    def initialize_points(self):
        self._embedding: np.ndarray = None
        self._binned_points: List[np.ndarray] = [self.empty_pointset for ic in range(self._n_point_bins)]
        self._points: np.ndarray = self.empty_pointset
        self._marker_points: List[np.ndarray] = None
        self._marker_pids: List[np.ndarray] = None

    def initialize_markers(self):
        if self._marker_points is None:
            nLabels = LabelsManager.instance().nLabels
            self._marker_points: List[np.ndarray] = [ self.empty_pointset for ic in range( nLabels ) ]
            self._marker_pids: List[np.ndarray] = [ self.empty_pids for ic in range( nLabels ) ]

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
        project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("points")
        reduced_data: xa.DataArray = project_dataset.reduction
#        reduced_data.attrs['dsid'] = project_dataset.attrs['dsid']
        print( f"UMAP init, init data shape = {reduced_data.shape}")
        self._embedding = ReductionManager.instance().umap_init( reduced_data, **kwargs  )
        self._points = self._embedding
        self.initialize_markers()

    def reembed(self, **kwargs ):
        t0 = time.time()
        self.clear_bins()
        self._embedding = ReductionManager.instance().umap_embedding( **kwargs )
        self.update_plot()
        print(f"PointCloudManager: completed embed in {time.time()-t0} sec\n\n\n")

    def update_plot( self, **kwargs ):
        self._points = kwargs.get( 'points', self._embedding )
        self._gui.point_sets = self.point_sets

    def on_selection(self, selection_event: Dict ):
        selection = selection_event['pids']
        self.update_markers(selection)

    def update_markers(self, pids: List[int]):
        self._marker_points[0] = self._embedding[ pids, : ]
        print( f"  ***** POINTS- mark_points[0], #pids = {len(pids)}")
        self.update_plot()

    def mark_points(self, pids: np.ndarray, cid: int = -1, update=False):
        from spectraclass.model.labels import LabelsManager
        print( f"PointCloudManager.mark_points: pids = {pids}, cid = {cid}")
        self.initialize_markers()
        lmgr = LabelsManager.instance()
        icid: int = cid if cid > 0 else lmgr.current_cid
        self.clear_pids( pids )
        self.clear_points(0)
        self._marker_pids[icid] = np.append( self._marker_pids[icid], pids )
        marked_points: np.ndarray = self._embedding[ self._marker_pids[icid], : ]
#        print( f"  ***** POINTS- mark_points[{icid}], #pids = {len(pids)}, #points = {marked_points.shape[0]}")
        self._marker_points[ icid ] = marked_points # np.concatenate(  [ self._marker_points[ icid ], marked_points ] )
        lmgr.addAction( "mark", "points", pids, icid )
        if update: self.update_plot()
        return lmgr.current_cid

    def clear_bins(self):
        for iC in range( 0, self._n_point_bins ):
            self._binned_points[iC] = self.empty_pointset

    def color_by_value( self, D: np.ndarray, base = 12.0, log_scale = False ):
        DM = ma.masked_invalid(D)
        N_log_bins = self._n_point_bins-1
        if log_scale:
            v1 = 0.9*DM.max(); v0 = 0.01*v1
            print(f" binned_points: v1 = {v1}, v0 = {v0}, base = {base}, D.shape = {DM.shape}")
            dmin, dmax = math.log(v0,base), math.log(v1,base)
            lspace: np.ndarray = np.logspace( dmin, dmax, N_log_bins )
            print(f"     ---> dmin = {dmin}, dmax = {dmax}, lspace = {lspace}")
        else:
            lspace: np.ndarray = np.linspace( DM.min(), DM.max(), N_log_bins )
        self._binned_points[0] = self._embedding[DM <= lspace[0]]
        print( f" binned_points[0]: size = {self._binned_points[0].shape[0]}, bin = (< {lspace[0]})")
        for iC in range(0,N_log_bins-1):
            mask: np.ndarray =  ( DM > lspace[iC] ) & ( DM <= lspace[iC+1] )
            self._binned_points[iC+1] = self._embedding[ mask ]
            print( f" binned_points[{iC+1}]: size = {self._binned_points[iC].shape[0]}, bin = [{lspace[iC]},{lspace[iC+1]}]")
        self._binned_points[-1] = self._embedding[ DM >  lspace[-1] ]
        print(f" binned_points[{self._n_point_bins-1}]: size = {self._binned_points[-1].shape[0]}, bin = (> {lspace[-1]})")
        LabelsManager.instance().addAction( "color", "points" )
        self.update_plot()

    def clear_pids(self, pids: np.ndarray, **kwargs):
        if self._marker_pids is not None:
            dpts = np.vectorize(lambda x: x in pids)
            for iC, marker_pids in enumerate( self._marker_pids ):
                if len( marker_pids ) > 0:
                    self._marker_pids[iC] = np.delete( self._marker_pids[iC], dpts(marker_pids) )
                    self._marker_points[iC] = self._embedding[self._marker_pids[iC], :] if len( self._marker_pids[iC] ) > 0 else self.empty_pointset

    def clear_points(self, icid: int, **kwargs ):
        if self._marker_pids is not None:
            pids = kwargs.get('pids', None )
            print( f"POINTS.clear: cid={icid}, pids={pids}")
            if pids is None:
                self._marker_points[icid] = self.empty_pointset
                self._marker_pids[icid] = self.empty_pids
            else:
                dpts = np.vectorize( lambda x: x in pids )
                dmask = dpts( self._marker_pids[icid] )
    #            print( f"clear_points.Mask: {self._marker_pids[icid]} -> {dmask}" )
                self._marker_pids[icid]  = np.delete( self._marker_pids[icid], dmask )
                self._marker_points[ icid ] = self._embedding[ self._marker_pids[icid], :] if len( self._marker_pids[icid] ) > 0 else self.empty_pointset
    #            print(f"clear_points: reduced marker_pids = {self._marker_pids[icid]} -> points = {self._marker_points[ icid ]}")

    @property
    def point_sets(self):
        self.initialize_markers()
        return [ self._points ] + self._binned_points + self._marker_points[::-1]

    def gui(self, **kwargs ):
        if self._gui is None:
            self.init_data()
            bin_colors = self.get_bin_colors("gist_rainbow") # self.get_bin_colors("jet",True)
            ptcolors = [ [1.0, 1.0, 1.0, 1.0], ] + bin_colors + LabelsManager.instance().colors[::-1]
            ptsizes = [1]*(self._n_point_bins+1) + [8]*LabelsManager.instance().nLabels
            self._gui = view( point_sets = self.point_sets, point_set_sizes=ptsizes, point_set_colors=ptcolors, background=[0,0,0] )
            self._gui.layout = { 'width': 'auto', 'flex': '1 1 auto' }
        return self._gui

    def refresh(self):
        self._gui = None
