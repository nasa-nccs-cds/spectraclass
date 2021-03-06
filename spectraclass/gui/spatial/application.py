import os, ipywidgets as ipw
from spectraclass.util.logs import LogManager, lgm
from spectraclass.application.controller import SpectraclassController
from spectraclass.model.base import Marker

class Spectraclass(SpectraclassController):

    def __init__(self):
        super(Spectraclass, self).__init__()
        self.set_parent_instances()

    def gui( self, embed: bool = False ):
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.gui.control import ActionsManager, am, ParametersManager, pm, UserFeedbackManager, ufm
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.gui.spatial.satellite import SatellitePlotManager, spm
        print( f"Initializing GUI using controller {str(self.__class__)}")

        self.set_spectraclass_theme()
        css_border = '1px solid blue'
        lgm().log(f"Creating app gui using class {str(self.__class__)}")

        plot_collapsibles = ipw.Accordion(children = [dm().gui(), pcm().gui(), spm().gui() ], layout=ipw.Layout(width='100%'))     #
        for iT, title in enumerate(['data', 'embedding', 'satellite']): plot_collapsibles.set_title(iT, title)
        plot_collapsibles.selected_index = 1
        plot = ipw.VBox([ ufm().gui(), plot_collapsibles ], layout=ipw.Layout( flex='1 0 700px' ), border=css_border )
        lgm().log("Created panel 1")

        control_collapsibles = ipw.Accordion(children=[gpm().gui(), pm().gui()], layout=ipw.Layout(width='100%'))  # , lm.gui()
        for iT, title in enumerate(['graph', 'controls']): control_collapsibles.set_title(iT, title)   # , 'logs'
        control_collapsibles.selected_index = 0
        control = ipw.VBox( [ am().gui(), mm().gui(), control_collapsibles ], layout=ipw.Layout( flex='0 0 700px'), border=css_border )
        lgm().log("Created panel 2")

        gui = ipw.HBox( [control, plot ], layout=ipw.Layout( width='100%' ) )
        if embed: self.embed()
        dm().save_config()
        lgm().log("Created app gui")
        return gui

    def mark(self):
        super(Spectraclass, self).mark()
        from spectraclass.gui.spatial.map import MapManager, mm
        mm().plot_markers_image()

    def clear(self):
        super(Spectraclass, self).clear()
        lgm().log( f"Spatial Spectraclass -> clear ")
        from spectraclass.gui.spatial.map import MapManager, mm
        mm().clearLabels()

    def undo_action(self):
        lgm().log(f"Spatial Spectraclass -> undo_action ")
        action = super(Spectraclass, self).undo_action()
        from spectraclass.gui.spatial.map import MapManager, mm
        if action is not None:
            if action.type == "mark":
                mm().plot_markers_image()

    def spread_selection(self, niters=1):
        from spectraclass.gui.spatial.map import MapManager, mm
        if super(Spectraclass, self).spread_selection()  is not None:
            mm().plot_markers_image()

    def add_marker(self, source: str, marker: Marker):
        from spectraclass.gui.spatial.map import MapManager, mm
        super(Spectraclass, self).add_marker( source, marker )
        lgm().log("spatial controller: add_marker")
        mm().plot_markers_image()








