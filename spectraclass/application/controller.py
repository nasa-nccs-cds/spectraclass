import os, ipywidgets as ipw
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled
from typing import List, Union, Tuple, Optional, Dict, Callable
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
        lgm().log(f"\n\nController[{self.__class__.__name__}] -> MARK ")
        lm().mark_points()
        pcm().update_marked_points()

    @exception_handled
    def clear(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, lm
        lgm().log(f"\n\nController[{self.__class__.__name__}] -> CLEAR ")
        lm().clearMarkers()
        pcm().clear()

    @exception_handled
    def embed(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.reduction.embedding import ReductionManager, rm
        lgm().log(f"\n\nController[{self.__class__.__name__}] -> EMBED ")
        embedding = rm().umap_embedding()
        pcm().reembed(embedding)

    @exception_handled
    def undo_action(self):
        from spectraclass.model.labels import LabelsManager, Action, lm
        lgm().log(f"\n\nController[{self.__class__.__name__}] -> UNDO ")
        action: Optional[Action] = lm().popAction()
        is_transient = self.process_action( action )
        if is_transient:
            action = lm().popAction()
            self.process_action(action)
        return action

    def process_action( self, action ) -> bool:
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.plot import PlotManager, gm
        is_transient = False
        lgm().log(f" UNDO action:  {action}")
        if action is not None:
            if action.type == "mark":
                m: Marker = action["marker"]
                if m.cid == 0: is_transient = True
                lgm().log(f" POP marker:  {m}")
                pcm().clear_pids(m.cid, m.pids)
                gm().plot_graph()
            elif action.type == "color":
                pcm().clear_bins()
            pcm().update_plot()
            lm().log_markers("post-undo")
        return is_transient

    @exception_handled
    def spread_selection(self, niters=1):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.plot import PlotManager, gm
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        lgm().log(f"\n\nController[{self.__class__.__name__}] -> SPREAD ")
        flow: ActivationFlow = afm().getActivationFlow()
        lm().log_markers("pre-spread")
        self._flow_class_map: np.ndarray = lm().labels_data().data
        catalog_pids = np.arange(0, self._flow_class_map.shape[0])
        pcm().clear_bins()
        converged = flow.spread(self._flow_class_map, niters)

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
            gm().plot_graph()
        lm().log_markers("post-spread")
        return converged

    @exception_handled
    def display_distance(self, niters=100):
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.points import PointCloudManager, pcm
        lgm().log(f"\n\nController[{self.__class__.__name__}] -> DISTANCE ")
        seed_points: xa.DataArray = lm().getSeedPointMask()
        flow: ActivationFlow = afm().getActivationFlow()
        if flow.spread( seed_points.data, niters ) is not None:
            pcm().color_by_value( flow.get_distances(), distance=True )

    @exception_handled
    def add_marker(self, source: str, marker: Marker):
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.plot import PlotManager, gm
        from spectraclass.gui.points import PointCloudManager, pcm
        lgm().log(f"\n\nController[{self.__class__.__name__}] -> ADD MARKER ")
        lm().addMarkerAction( "app", marker )
        pids = marker.pids[np.where(marker.pids >= 0)]
        gm().plot_graph(pids)
        pcm().update_marked_points(marker.cid)
        lm().log_markers("post-add_marker")

    @exception_handled
    def color_pointcloud( self, color_data: np.ndarray = None, **kwargs ):
        from spectraclass.gui.points import PointCloudManager, pcm
        pcm().color_by_value( color_data, **kwargs )







