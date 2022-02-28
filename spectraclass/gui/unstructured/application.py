import os, ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm
from spectraclass.application.controller import SpectraclassController
from spectraclass.gui.spatial.widgets.markers import Marker

class Spectraclass(SpectraclassController):

    def __init__(self):
        super(Spectraclass, self).__init__()
        self.set_parent_instances()
        self._gui = None

    def process_menubar_action(self, mname, dname, op, b ):
        print(f" process_menubar_action.on_value_change: {mname}.{dname} -> {op}")

    def gui( self, **kwargs ):
        if self._gui is None:
            self.create_gui( **kwargs )
        return self._gui

    def create_gui(self, **kwargs):
        from bokeh.io import output_notebook
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.gui.points3js import PointCloudManager, pcm
        from spectraclass.gui.unstructured.table import tm
        from spectraclass.gui.control import ActionsManager, am, ParametersManager, pm, UserFeedbackManager, ufm
        from spectraclass.data.base import DataManager, dm
        lgm().log( f"Creating the Spectraclass[{id(self)}] gui")
        output_notebook()
        embed: bool = kwargs.pop('embed',False)
        self.set_spectraclass_theme()
        css_border = '1px solid blue'
        collapsibles = ipw.Accordion( children = [ dm().gui(), pcm().gui() ], layout=ipw.Layout(width='100%'))
        for iT, title in enumerate(['data', 'embedding']): collapsibles.set_title(iT, title)
        collapsibles.selected_index = 1
        plot = ipw.VBox([ ufm().gui(), collapsibles ], layout=ipw.Layout( flex='1 0 700px' ), border=css_border )
        control = ipw.VBox([am().gui(), tm().gui(), gpm().gui()], layout=ipw.Layout(flex='0 0 700px'), border=css_border)
        self._gui = ipw.HBox( [control, plot ], layout=ipw.Layout( width='100%' ) )
        if embed: self.embed()

    def mark(self):
        super(Spectraclass, self).mark()
    #    lgm().log(f"      *UNSTRUCTURED CONTROLLER -> MARK ")
        from spectraclass.gui.unstructured.table import tm
        tm().mark_points()

    def clear(self):
        from spectraclass.gui.unstructured.table import tm
        super(Spectraclass, self).clear()
        lgm().log( f"      *UNSTRUCTURED CONTROLLER -> CLEAR ")

    def undo_action(self):
        from spectraclass.gui.unstructured.table import tm
        super(Spectraclass, self).undo_action()
        lgm().log(f"      *UNSTRUCTURED CONTROLLER -> UNDO ")

    def propagate_selection(self, niters=1):
        from spectraclass.gui.unstructured.table import tm
        super(Spectraclass, self).propagate_selection()
        lgm().log(f"      *UNSTRUCTURED CONTROLLER -> SPREAD ")

    def add_marker(self, source: str, marker: Marker):
        from spectraclass.gui.unstructured.table import tm
        super(Spectraclass, self).add_marker( source, marker )
        lgm().log(f"      *UNSTRUCTURED CONTROLLER -> ADD_MARKER ")






