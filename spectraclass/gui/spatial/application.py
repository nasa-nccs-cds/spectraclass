import os, ipywidgets as ipw
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
        from spectraclass.gui.graph import GraphManager, gm
        from spectraclass.gui.points import PointCloudManager, pcm
        from spectraclass.gui.control import ActionsManager, am, ControlsManager, cm, UserFeedbackManager, ufm
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.gui.spatial.google import GooglePlotManager, gpm

        self.set_spectraclass_theme()
        css_border = '1px solid blue'
        collapsibles = ipw.Accordion( children = [ cm().gui(), pcm().gui(), gpm().gui() ], layout=ipw.Layout( width='100%' ) )
        for iT, title in enumerate(['controls', 'embedding', 'satellite']): collapsibles.set_title(iT, title)
        collapsibles.selected_index = 1
        plot = ipw.VBox([ ufm().gui(), collapsibles ], layout=ipw.Layout( flex='1 0 700px' ), border=css_border )
        control = ipw.VBox( [ am().gui(), mm().gui(), gm().gui() ], layout=ipw.Layout( flex='0 0 700px'), border=css_border )
        gui = ipw.HBox( [control, plot ], layout=ipw.Layout( width='100%' ) )
        if embed: ActionsManager.instance().embed()
        return gui

    def show_gpu_usage(self):
        os.system("nvidia-smi")





