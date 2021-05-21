import os, ipywidgets as ipw
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.model.base import Marker
import numpy as np
import xarray as xa

def app():
    from spectraclass.data.base import DataManager, dm
    rv = dm().app()
    return rv

class SpectraclassController(SCSingletonConfigurable):

    HOME = os.path.dirname( os.path.dirname( os.path.dirname(os.path.realpath(__file__)) ) )
    custom_theme = False

    def __init__(self):
        super(SpectraclassController, self).__init__()

    # def set_controller_instance(self):  # cls.__bases__
    #     assert SpectraclassController._instance is None, "Error, SpectraclassController cannot be instantiated"
    #     SpectraclassController._instance = self
    #     SpectraclassController._instantiated = self.__class__

    def process_menubar_action(self, mname, dname, op, b ):
        print(f" process_menubar_action.on_value_change: {mname}.{dname} -> {op}")

    def show_gpu_usage(self):
        os.system("nvidia-smi")

    @property
    def color_map(self) -> str:
        from spectraclass.gui.points import PointCloudManager, pcm
        return pcm().color_map

    def update_current_class(self, iclass: int ):
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        pids = lm().getPids( iclass )
        gpm().plot_graph(pids)

    @classmethod
    def set_spectraclass_theme(cls):
        from IPython.display import display, HTML
        if cls.custom_theme:
            theme_file = os.path.join( cls.HOME, "themes", "spectraclass.css" )
            with open( theme_file ) as f:
                css = f.read().replace(';', ' !important;')
            display(HTML('<style type="text/css">%s</style>Customized changes loaded.' % css))

    def gui( self, embed: bool = False ):
        raise NotImplementedError()

    @exception_handled
    def mark(self):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.points import PointCloudManager, pcm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> MARK ")
        lm().mark_points()
        pcm().update_marked_points()

    @exception_handled
    def clear(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.satellite import SatellitePlotManager, spm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> CLEAR ")
        if lm().clearMarkers():
            gpm().clear()
            pcm().clear()
        else:
            spm().clear_overlay_image()


    @exception_handled
    def embed(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.reduction.embedding import ReductionManager, rm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> EMBED ")
        ufm().show( "Computing 3D embedding")
        embedding = rm().umap_embedding()
        pcm().reembed(embedding)
        ufm().clear()

    @exception_handled
    def undo_action(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        lgm().log(f"                  ----> Controller[{self.__class__.__name__}] -> UNDO ")
        while True:
            action: Optional[Action] = lm().popAction()
            is_transient = self.process_undo( action )
            if not is_transient: break
        pcm().update_plot()
        gpm().plot_graph()
        return action

    def process_undo( self, action ) -> bool:
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.spatial.satellite import SatellitePlotManager, spm
        from spectraclass.gui.spatial.map import MapManager, mm
        is_transient = False
        if action is not None:
            lgm().log(f" UNDO action:  {action}")
            if action.type == "mark":
                m: Marker = action["marker"]
                if m.cid == 0: is_transient = True
                lgm().log(f" POP marker:  {m}")
                pcm().clear_pids(m.cid, m.pids)
            elif action.type == "color":
                pcm().clear_bins()
                mm().clear_overlay_image()
                spm().clear_overlay_image()
            lm().log_markers("post-undo")
        return is_transient

    @exception_handled
    def classify(self) -> xa.DataArray:
        from spectraclass.learn.base import ClassificationManager, cm
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.spatial.satellite import SatellitePlotManager, spm

        embedding: xa.DataArray = dm().getModelData()
        classification: xa.DataArray = cm().apply_classification( embedding )
        pcm().color_by_index( classification.data, lm().colors )
        overlay_image = classification.data.reshape( mm().image_template.shape )
        mm().plot_overlay_image( overlay_image )
        spm().plot_overlay_image( mm().image_template.copy( data=overlay_image ), mm().overlay_alpha )
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
        pcm().clear_bins()
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
                        pcm().update_marked_points(cid)
            gpm().plot_graph()
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
            pcm().color_by_value( flow.get_distances(), distance=True )
            lm().addAction("color", "points")

    @exception_handled
    def add_marker(self, source: str, marker: Marker):
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.gui.points import PointCloudManager, pcm
        lm().addMarkerAction( "app", marker )
        pids = marker.pids[np.where(marker.pids >= 0)]
        lgm().log(f"  ----> Controller[{self.__class__.__name__}] -> ADD MARKER, pids = {pids} ")
        gpm().plot_graph(pids)
        pcm().update_marked_points(marker.cid)
        lm().log_markers("post-add_marker")

    @exception_handled
    def color_pointcloud( self, color_data: np.ndarray = None, **kwargs ):
        from spectraclass.gui.points import PointCloudManager, pcm
        pcm().color_by_value( color_data, **kwargs )







