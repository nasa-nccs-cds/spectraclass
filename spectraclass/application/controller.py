import os, ipywidgets as ipw
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.gui.spatial.widgets.markers import Marker
import numpy as np
import xarray as xa

class ActionEvent(object):

    def __init__( self, type: str ):
        super(ActionEvent, self).__init__()
        self._type = type

class LabelEvent(ActionEvent):

    def __init__( self, type: str, label_map: np.ndarray ):
        super(LabelEvent, self).__init__( type )
        self._label_map = label_map

    @property
    def label_map(self):
        return self._label_map

class SpectraclassController(SCSingletonConfigurable):

    HOME = os.path.dirname( os.path.dirname( os.path.dirname(os.path.realpath(__file__)) ) )
    custom_theme = False

    def __init__(self):
        super(SpectraclassController, self).__init__()
        self._action_events = []

    @classmethod
    def set_spectraclass_theme(cls):
        from IPython.display import display, HTML
        if cls.custom_theme:
            theme_file = os.path.join( cls.HOME, "themes", "spectraclass.css" )
            with open( theme_file ) as f:
                css = f.read().replace(';', ' !important;')
            display(HTML('<style type="text/css">%s</style>Customized changes loaded.' % css))

    def addActionEvent(self, event: ActionEvent ):
        self._action_events.append( event )

    def popActionEvent(self) -> ActionEvent:
        return self._action_events.pop()

    def lastActionEvent(self) -> ActionEvent:
        return self._action_events[-1]

    def process_menubar_action(self, mname, dname, op, b ):
        print(f" process_menubar_action.on_value_change: {mname}.{dname} -> {op}")

    def show_gpu_usage(self):
        os.system("nvidia-smi")

    @property
    def color_map(self) -> str:
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        return pcm().color_map

    @exception_handled
    def update_current_class(self, iclass: int ):
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        gids = lm().getGids( iclass )
        gpm().plot_graph( Marker( "marker", gids, iclass ) )
        mm().set_region_class( iclass )

    def gui( self, **kwargs ):
        raise NotImplementedError()

    @exception_handled
    def mark(self):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
    #    lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> MARK ")

    @exception_handled
    def mask(self):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> MASK ")
#        mm().create_mask( lm().current_cid )

    @exception_handled
    def clear(self):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> CLEAR ")
        lm().clearMarkers()
        mm().clearMarkers()
        gpm().clear()
        pcm().clear()
        mm().plot_labels_image()

    @exception_handled
    def embed(self):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        from spectraclass.reduction.embedding import ReductionManager, rm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> EMBED ")
        ufm().show( "Computing 3D embedding" )
        mm().update_pcm()
        embedding = rm().umap_embedding()
        pcm().update_plot( points=embedding )
        ufm().clear()

    @exception_handled
    def undo_action(self):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        action: Optional[Action] = lm().popAction()
        if action is not None:
            lgm().log(f"  ----> Controller[{self.__class__.__name__}] -> UNDO: {action} ")
            if action.type == "spread":
                marker = lm().popMarker( "labels" )
                lgm().log(f"undo_action-> pop marker: {marker}")
                mm().plot_labels_image( lm().get_label_map() )
            if action.type == "classify":
                mm().plot_labels_image( lm().get_label_map() )

    @log_timing
    def cluster(self):
        from spectraclass.learn.cluster.manager import clm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.spatial.map import MapManager, mm
        ufm().show(f"Creating clusters using {clm().mid}  ")
        cluster_input: xa.DataArray = dm().getModelData()
        cluster_image: xa.DataArray = clm().cluster( cluster_input )
        mm().plot_cluster_image( cluster_image )
        ufm().show(f"Clustering completed")

    def learn(self):
        from spectraclass.learn.manager import ClassificationManager, cm
        ufm().show("Learning Classification Mapping... ")
        cm().learn_classification()
        ufm().show( "Classification Mapping learned" )

    # @log_timing
    # def learn(self):
    #     from spectraclass.learn.manager import ClassificationManager, cm
    #     from spectraclass.data.base import DataManager, dm
    #     from spectraclass.model.labels import LabelsManager, Action, lm
    #     ufm().show("Learning Classification Mapping... ")
    #     lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> LEARN ")
    #     embedding: xa.DataArray = dm().getModelData()
    #     labels_data: xa.DataArray = lm().getLabelsArray()
    #     labels_mask = (labels_data > 0)
    #     filtered_labels: np.ndarray = labels_data.where(labels_mask, drop=True).astype(np.int32).values
    #     filtered_point_data: np.ndarray = embedding.where(labels_mask, drop=True).values
    #     lgm().log(f"SHAPES--> embedding: {embedding.shape}, labels_data: {labels_data.shape}, filtered_labels: {filtered_labels.shape}, filtered_point_data: {filtered_point_data.shape}" )
    #     cm().learn_classification( filtered_point_data, filtered_labels )
    #     ufm().show( "Classification Mapping learned" )

    @log_timing
    def propagate_selection(self, niters=1):
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.spatial.tile.tile import Block, Tile
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        block: Block = tm().getBlock()
        ufm().show("Generalizing markers")
        flow: ActivationFlow = afm().getActivationFlow()
        lm().log_markers("pre-spread")
        self._flow_class_map: np.ndarray = lm().getLabelsArray().data
        catalog_pids = np.arange(0, self._flow_class_map.shape[0])
        lgm().log(f"SPREAD: flow_class_map shape={self._flow_class_map.shape}, catalog_pids shape={catalog_pids.shape} ")
        converged = flow.spread( self._flow_class_map, niters )

        if converged is not None:
            self._flow_class_map = flow.get_classes()
            all_classes = ( lm().current_cid == 0 )
            for cid, label in enumerate( lm().labels ):
                if all_classes or ( lm().current_cid == cid ):
                    new_pids: np.ndarray = catalog_pids[ self._flow_class_map == cid ]
                    if new_pids.size > 0:
                        lgm().log(f" @@@ spread_selection: cid={cid}, label={label}, #new_indices={len(new_pids)}" )
                        gids: np.ndarray = block.pids2gids( new_pids )
                        lm().mark_points( gids, cid, "labels" )
                        lm().addAction( "spread", "application", cid=cid )
            mm().plot_labels_image( lm().get_label_map() )
        lm().log_markers("post-spread")
        ufm().show("Marker generalization complete")
        return converged

    @exception_handled
    def display_distance(self, niters=100):
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        ufm().show("Coloring by Distance")
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> DISTANCE ")
        seed_points: xa.DataArray = lm().getSeedPointMask()
        flow: ActivationFlow = afm().getActivationFlow()
        if flow.spread( seed_points.data, niters, bidirectional=True ) is not None:
            pcm().color_by_value( flow.get_distances(), distance=True )
            ufm().show("Done Coloring by Distance")

    @exception_handled
    def add_marker(self, marker: Marker):
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        if marker is not None:
            lgm().log( f" #APP-> Add Marker[{marker.cid}] ")
            lm().addMarker( marker )
            gpm().plot_graph( marker )
            pcm().addMarker(marker)

    @exception_handled
    def remove_marker(self, marker: Marker):
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        if marker is not None:
            lm().clearMarker( marker )
            gpm().remove_marker( marker )
            pcm().deleteMarkers(marker.gids.tolist())

    def get_marked_pids(self) -> Dict[int,Set[int]]:
        from spectraclass.model.labels import LabelsManager, Action, lm
        marked_pids = {}
        for marker in lm().markers:
            new_pids = marker.gids[ np.where(marker.gids >= 0)].tolist()
            current_pids = marked_pids.get( marker.cid, set() )
            marked_pids[ marker.cid ] = current_pids.union( set(new_pids) )
        return marked_pids

    @exception_handled
    def color_pointcloud( self, color_data: np.ndarray = None, **kwargs ):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        pcm().color_by_value( color_data, **kwargs )


def app() -> SpectraclassController:
    from spectraclass.data.base import DataManager, dm
    rv = dm().app()
    return rv








