import os, ipywidgets as ipw
from spectraclass.util.logs import LogManager, lgm
from spectraclass.application.controller import SpectraclassController
from spectraclass.model.base import Marker

class Spectraclass(SpectraclassController):

    def __init__(self):
        super(Spectraclass, self).__init__()
        self.set_parent_instances()

    def gui( self, embed: bool = False ):
        from spectraclass.gui.plot import PlotManager, gm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.gui.control import ActionsManager, am, ControlsManager, cm, UserFeedbackManager, ufm
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.gui.spatial.google import GooglePlotManager, gpm

        self.set_spectraclass_theme()
        css_border = '1px solid blue'
        collapsibles = ipw.Accordion( children = [ cm().gui(), pcm().gui(), gpm().gui() ], layout=ipw.Layout( width='100%' ) )  # , lm.gui()
        for iT, title in enumerate(['controls', 'embedding', 'satellite']): collapsibles.set_title(iT, title)    # , 'logs'
        collapsibles.selected_index = 1
        plot = ipw.VBox([ ufm().gui(), collapsibles ], layout=ipw.Layout( flex='1 0 700px' ), border=css_border )
        control = ipw.VBox( [ am().gui(), mm().gui(), gm().gui() ], layout=ipw.Layout( flex='0 0 700px'), border=css_border )
        gui = ipw.HBox( [control, plot ], layout=ipw.Layout( width='100%' ) )
        if embed: self.embed()
        dm().save_config()
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








