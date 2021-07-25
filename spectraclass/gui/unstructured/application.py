import os, ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm
from spectraclass.application.controller import SpectraclassController
from spectraclass.model.base import  Marker

class Spectraclass(SpectraclassController):

    def __init__(self):
        super(Spectraclass, self).__init__()
        self.set_parent_instances()

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
        from bokeh.io import output_notebook
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.gui.unstructured.table import tm
        from spectraclass.gui.control import ActionsManager, am, ParametersManager, pm, UserFeedbackManager, ufm
        from spectraclass.data.base import DataManager, dm
        output_notebook()

        self.set_spectraclass_theme()
        css_border = '1px solid blue'
        collapsibles = ipw.Accordion( children = [ dm().gui(), pcm().gui() ], layout=ipw.Layout(width='100%'))
        for iT, title in enumerate(['data', 'embedding']): collapsibles.set_title(iT, title)
        collapsibles.selected_index = 1
        plot = ipw.VBox([ ufm().gui(), collapsibles ], layout=ipw.Layout( flex='1 0 700px' ), border=css_border )
        control = ipw.VBox([am().gui(), tm().gui(), gpm().gui()], layout=ipw.Layout(flex='0 0 700px'), border=css_border)
        gui = ipw.HBox( [control, plot ], layout=ipw.Layout( width='100%' ) )
        if embed: self.embed()
        dm().save_config()
        return gui

    def mark(self):
        super(Spectraclass, self).mark()
        lgm().log(f"      *UNSTRUCTURED CONTROLLER -> MARK ")
        from spectraclass.gui.unstructured.table import tm
        tm().update_selection("mark")

    def clear(self):
        from spectraclass.gui.unstructured.table import tm
        super(Spectraclass, self).clear()
        lgm().log( f"      *UNSTRUCTURED CONTROLLER -> CLEAR ")
        tm().update_selection("clear")

    def undo_action(self):
        from spectraclass.gui.unstructured.table import tm
        super(Spectraclass, self).undo_action()
        lgm().log(f"      *UNSTRUCTURED CONTROLLER -> UNDO ")
        tm().update_selection("undo")

    def spread_selection(self, niters=1):
        from spectraclass.gui.unstructured.table import tm
        super(Spectraclass, self).spread_selection()
        lgm().log(f"      *UNSTRUCTURED CONTROLLER -> SPREAD ")
        tm().update_selection("spread")

    def add_marker(self, source: str, marker: Marker):
        from spectraclass.gui.unstructured.table import tm
        super(Spectraclass, self).add_marker( source, marker )
        lgm().log(f"      *UNSTRUCTURED CONTROLLER -> ADD_MARKER ")
        tm().update_selection("add")





