import os, ipywidgets as ipw
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm
from spectraclass.model.base import Marker
import numpy as np
import xarray as xa

def app(): return SpectraclassController.instance()

class SpectraclassController(SCSingletonConfigurable):

    HOME = os.path.dirname( os.path.dirname( os.path.dirname(os.path.realpath(__file__)) ) )
    custom_theme = False

    def __init__(self):
        super(SpectraclassController, self).__init__()

    def set_controller_instance(self):
        assert SpectraclassController._instance is None, "Error, SpectraclassController cannot be instantiated"
        SpectraclassController._instance = self
        SpectraclassController._instantiated = self.__class__

    def process_menubar_action(self, mname, dname, op, b ):
        print(f" process_menubar_action.on_value_change: {mname}.{dname} -> {op}")

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

    def show_gpu_usage(self):
        os.system("nvidia-smi")

    def mark(self):
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.points import PointCloudManager, pcm
        lm().mark_points()
        pcm().update_marked_points()

    def clear(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, lm
        lgm().log(f"Controller[{self.__class__.__name__}] -> clear ")
        lm().clearMarkers()
        pcm().clear()

    def embed(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.reduction.embedding import ReductionManager, rm
        embedding = rm().umap_embedding()
        pcm().reembed(embedding)

    def undo_action(self):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, Action, lm
        lgm().log(f"Controller[{self.__class__.__name__}] -> undo_action ")
        action: Action = lm().popAction()
        lgm().log( f" UNDO action:  {action}" )
        if action is not None:
            if action.type == "mark":
                m: Marker = lm().popMarker()
                lgm().log(f" POP marker:  {m}")
                pcm().clear_pids( m.cid, m.pids )
            elif action.type == "color":
                pcm().clear_bins()
        pcm().update_plot( )
        return action

    def spread_selection(self, niters=1):
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        try:
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
                            lm().mark_points( new_indices, cid )
                            pcm().update_marked_points(cid)
            lm().log_markers("post-spread")
            return converged
        except Exception:
            lgm().exception( "Error in 'spread_selection'")

    def display_distance(self, niters=100):
        from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
        from spectraclass.model.labels import LabelsManager, Action, lm
        from spectraclass.gui.points import PointCloudManager, pcm
        seed_points: xa.DataArray = lm().getSeedPointMask()
        flow: ActivationFlow = afm().getActivationFlow()
        if flow.spread( seed_points.data, niters ) is not None:
            pcm().color_by_value( flow.get_distances(), distance=True )






