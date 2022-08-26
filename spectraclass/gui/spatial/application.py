from spectraclass.util.logs import LogManager, lgm
import ipywidgets as ipw
import matplotlib.pyplot as plt
from spectraclass.application.controller import SpectraclassController
from spectraclass.gui.spatial.widgets.markers import Marker

class Spectraclass(SpectraclassController):

    def __init__(self):
        super(Spectraclass, self).__init__()
        self.set_parent_instances()

    def gui( self, **kwargs ):
        try:
            from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
            from spectraclass.data.base import DataManager, dm
            from spectraclass.gui.control import ActionsManager, am, ParametersManager, pm, UserFeedbackManager, ufm
            from spectraclass.gui.spatial.map import MapManager, mm
            print( f"Initializing GUI using controller {str(self.__class__)}")
            basemap: bool = kwargs.pop('basemap',True)
            embed: bool = kwargs.pop('embed',False)
            self.set_spectraclass_theme()
            css_border = '1px solid blue'
            plot_collapsibles = ipw.Accordion( children = [ dm().gui(), pm().gui(), self.create_viz_tabs() ], layout=ipw.Layout(width='100%'))
            for iT, title in enumerate(['data', 'controls', 'visualization' ]): plot_collapsibles.set_title(iT, title)
            plot_collapsibles.selected_index = 1
            plot = ipw.VBox([ ufm().gui(), plot_collapsibles, gpm().gui() ], layout=ipw.Layout( flex='1 0 700px' ), border=css_border )
            smap = mm().gui( basemap=basemap )
            control = ipw.VBox( [ am().gui(), smap ], layout=ipw.Layout( flex='0 0 700px'), border=css_border )
            gui = ipw.HBox( [control, plot ], layout=ipw.Layout( width='100%' ) )
            if embed: self.embed()
            return gui
        except Exception as err:
            lgm().exception( "Initialization error")
            print( f"\n ****** Initialization error, check log file for details: {lgm().log_file} ****** \n")

    def create_viz_tabs(self):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        from spectraclass.gui.custom.visuals import cviz
        wTab = ipw.Tab()
        tabNames = [ "embedding" ]
        children = [ pcm().gui() ]
        for vname in cviz().names: tabNames.append(vname)
        for viz in cviz().guis: children.append(viz)
        for iT, title in enumerate(tabNames):
            wTab.set_title(iT, title)
        wTab.children = children
        return wTab

    def mark(self):
        super(Spectraclass, self).mark()
        from spectraclass.gui.spatial.map import MapManager, mm
        mm().plot_markers_image()

    def clear(self):
        super(Spectraclass, self).clear()
        lgm().log( f"Spatial Spectraclass -> clear ")
        from spectraclass.gui.spatial.map import MapManager, mm
        mm().clearLabels()
        mm().init_cluster_image()

    def undo_action(self):
        lgm().log(f"Spatial Spectraclass -> undo_action ")
        action = super(Spectraclass, self).undo_action()
        from spectraclass.gui.spatial.map import MapManager, mm
        if action is not None:
            if action.type == "mark":
                mm().plot_markers_image()

    def propagate_selection(self, niters=1):
        from spectraclass.gui.spatial.map import MapManager, mm
        if super(Spectraclass, self).propagate_selection()  is not None:
            mm().plot_markers_image()

    def add_marker(self, source: str, marker: Marker):
        from spectraclass.gui.spatial.map import MapManager, mm
        super(Spectraclass, self).add_marker( source, marker )
        lgm().log("spatial controller: add_marker")
        mm().plot_markers_image()








