import os, ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.model.base import SCSingletonConfigurable

class Spectraclass(SCSingletonConfigurable):

    HOME = os.path.dirname( os.path.dirname( os.path.dirname(os.path.realpath(__file__)) ) )
    custom_theme = False

    def __init__(self):
        super(Spectraclass, self).__init__()

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
        if embed: ActionsManager.instance().embed()
        dm().save_config()
        return gui

    def show_gpu_usage(self):
        os.system("nvidia-smi")





