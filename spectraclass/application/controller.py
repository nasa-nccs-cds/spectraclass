import os, ipywidgets as ipw
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled
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
        self.pcm_active = False

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
        from spectraclass.gui.points import PointCloudManager, pcm
        if self.pcm_active: return pcm().color_map

    @exception_handled
    def update_current_class(self, iclass: int ):
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        pids = lm().getPids( iclass )
        gpm().plot_graph( Marker( pids, iclass ) )
        mm().set_region_class( iclass )

    @classmethod
    def set_spectraclass_theme(cls):
        from IPython.display import display, HTML
        if cls.custom_theme:
            theme_file = os.path.join( cls.HOME, "themes", "spectraclass.css" )
            with open( theme_file ) as f:
                css = f.read().replace(';', ' !important;')
            display(HTML('<style type="text/css">%s</style>Customized changes loaded.' % css))

    def gui( self, **kwargs ):
        raise NotImplementedError()

    @exception_handled
    def mark(self):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.points import PointCloudManager, pcm
    #    lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> MARK ")

    @exception_handled
    def mask(self):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> MASK ")
        mm().create_mask( lm().current_cid )

    @exception_handled
    def clear(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> CLEAR ")
        lm().clearMarkers()
        gpm().clear()
        if self.pcm_active: pcm().clear()

    @exception_handled
    def embed(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.reduction.embedding import ReductionManager, rm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> EMBED ")
        ufm().show( "Computing 3D embedding")
        embedding = rm().umap_embedding()
        if self.pcm_active: pcm().reembed(embedding)
        ufm().clear()

    @exception_handled
    def undo_action(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> UNDO ")


    @exception_handled
    def classify(self) -> xa.DataArray:
        from spectraclass.learn.base import ClassificationManager, cm
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.model.labels import LabelsManager, Action, lm

        embedding: xa.DataArray = dm().getModelData()
        classification: xa.DataArray = cm().apply_classification( embedding )
        if self.pcm_active: pcm().color_by_index( classification.data, lm().colors )
        overlay_image = classification.data.reshape( mm().image_template.shape )
        mm().plot_overlay_image( overlay_image )
       # spm().plot_overlay_image( mm().image_template.copy( data=labels_image ), mm().overlay_alpha )
        lm().addAction("color", "points")
        return classification

    @exception_handled
    def learn(self):
        from spectraclass.learn.base import ClassificationManager, cm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.model.labels import LabelsManager, Action, lm
        embedding: xa.DataArray = dm().getModelData()
        labels_data: xa.DataArray = lm().labels_data()
        labels_mask = (labels_data > 0)
        filtered_labels: np.ndarray = labels_data.where(labels_mask, drop=True).astype(np.int32).values
        filtered_point_data: np.ndarray = embedding.where(labels_mask, drop=True).values
        cm().learn_classification( filtered_point_data, filtered_labels )

    @exception_handled
    def spread_selection(self, niters=1):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> SPREAD ")
        flow: ActivationFlow = afm().getActivationFlow()
        lm().log_markers("pre-spread")
        self._flow_class_map: np.ndarray = lm().labels_data().data
        catalog_pids = np.arange(0, self._flow_class_map.shape[0])
        if self.pcm_active: pcm().clear_bins()
        converged = flow.spread( self._flow_class_map, niters )

        if converged is not None:
            self._flow_class_map = flow.get_classes()
            all_classes = ( lm().current_cid == 0 )
            for cid, label in enumerate( lm().labels ):
                if all_classes or ( lm().current_cid == cid ):
                    new_indices: np.ndarray = catalog_pids[ self._flow_class_map == cid ]
                    if new_indices.size > 0:
                        lgm().log(f" @@@ spread_selection: cid={cid}, label={label}, new_indices={new_indices}" )
                        lm().mark_points( new_indices, cid )
                        if self.pcm_active: pcm().update_marked_points(cid)
 #           gpm().plot_graph()
        lm().log_markers("post-spread")
        return converged

    @exception_handled
    def display_distance(self, niters=100):
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.points import PointCloudManager, pcm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> DISTANCE ")
        seed_points: xa.DataArray = lm().getSeedPointMask()
        flow: ActivationFlow = afm().getActivationFlow()
        if flow.spread( seed_points.data, niters, bidirectional=True ) is not None:
            if self.pcm_active: pcm().color_by_value( flow.get_distances(), distance=True )
            lm().addAction("color", "points")

    @exception_handled
    def add_marker(self, source: str, marker: Marker):
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.gui.points import PointCloudManager, pcm
        lm().addMarkerAction( "app", marker )
        gpm().plot_graph( marker )
        if self.pcm_active: pcm().update_marked_points(marker.cid)
        lm().log_markers("post-add_marker")

    def get_marked_pids(self) -> Dict[int,Set[int]]:
        from spectraclass.model.labels import LabelsManager, Action, lm
        marked_pids = {}
        for marker in lm().markers:
            new_pids = marker.pids[ np.where(marker.pids >= 0) ].tolist()
            current_pids = marked_pids.get( marker.cid, set() )
            marked_pids[ marker.cid ] = current_pids.union( set(new_pids) )
        return marked_pids

    @exception_handled
    def color_pointcloud( self, color_data: np.ndarray = None, **kwargs ):
        from spectraclass.gui.points import PointCloudManager, pcm
        if self.pcm_active: pcm().color_by_value( color_data, **kwargs )


def app() -> SpectraclassController:
    from spectraclass.data.base import DataManager, dm
    rv = dm().app()
    return rv








