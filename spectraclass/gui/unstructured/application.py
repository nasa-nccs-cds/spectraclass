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
        from spectraclass.gui.plot import PlotManager
        from spectraclass.gui.points import PointCloudManager
        from spectraclass.gui.unstructured.table import TableManager
        from spectraclass.gui.control import ActionsManager
        from spectraclass.application.controller import app
        from spectraclass.data.base import DataManager, dm

        self.set_spectraclass_theme()
        css_border = '1px solid blue'

        tableManager = TableManager.instance()
        graphManager = PlotManager.instance()
        pointCloudManager = PointCloudManager.instance()

        table = tableManager.gui()
        graph = graphManager.gui()
        points = pointCloudManager.instance().gui()

        tableManager.add_selection_listerner(graphManager.on_selection)
        tableManager.add_selection_listerner(pointCloudManager.on_selection)
        actionsPanel = ActionsManager.instance().gui()

        control = ipw.VBox([actionsPanel, table], layout=ipw.Layout( flex='0 0 600px', border=css_border) )
        plot = ipw.VBox([points, graph], layout=ipw.Layout( flex='1 1 auto', border=css_border) )
        gui = ipw.HBox([control, plot])
        if embed: app().embed()
        dm().save_config()
        return gui

    def mark(self):
        from spectraclass.gui.unstructured.table import TableManager, tm
        super(Spectraclass, self).mark()
        tm().mark_selection()

    def clear(self):
        from spectraclass.gui.unstructured.table import TableManager, tm
        super(Spectraclass, self).clear()
        lgm().log( f"Spatial Spectraclass -> clear ")
        tm().clear_current_class()

    def undo_action(self):
        from spectraclass.gui.unstructured.table import TableManager, tm
        lgm().log(f"Spatial Spectraclass -> undo_action ")
        action = super(Spectraclass, self).undo_action()
        tm().undo_action()

    def spread_selection(self, niters=1):
        from spectraclass.gui.unstructured.table import TableManager, tm
        if super(Spectraclass, self).spread_selection()  is not None:
            tm().spread_selection()

    def add_marker(self, marker: Marker):
        from spectraclass.gui.unstructured.table import TableManager, tm
        super(Spectraclass, self).add_marker(marker)
        tm().mark_selection()





