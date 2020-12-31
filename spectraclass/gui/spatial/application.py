import os, ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.model.base import SCConfigurable

class Spectraclass(tlc.SingletonConfigurable, SCConfigurable):

    HOME = os.path.dirname( os.path.dirname( os.path.dirname(os.path.realpath(__file__)) ) )
    custom_theme = False

    def __init__(self):
        super(Spectraclass, self).__init__()

    def initialize( self, name: str, mode: str ) -> "DataManager":
        dm = self.configure(name,mode)
        self.save_config()

    def configure( self, name: str, mode: str ) -> "DataManager":
        from spectraclass.data.base import DataManager
        dm = DataManager.initialize(name,mode)
        cfg_file = DataManager.instance().config_file()
        from traitlets.config.loader import load_pyconfig_files
        if os.path.isfile(cfg_file):
            (dir, fname) = os.path.split(cfg_file)
            config_files = [ 'configuration.py', fname ]
            print(f"Loading config files: {config_files} from dir {dir}")
            config = load_pyconfig_files( config_files, dir )
            for clss in self.config_classes:
                instance = clss.instance()
                print( f"Configuring instance {instance.__class__.__name__}")
                instance.update_config(config)
        else:
            print( f"Configuration error: '{cfg_file}' is not a file.")
        return dm

    def save_config( self ):
        from spectraclass.data.base import DataManager
        conf_dict = self.generate_config_file()
        globals = conf_dict.pop( 'global', "" )
        for mode, mode_conf_txt in conf_dict.items():
            cfg_file = os.path.realpath( DataManager.instance().config_file(mode) )
            os.makedirs( os.path.dirname(cfg_file), exist_ok=True )
            with open( cfg_file, "w" ) as cfile_handle:
                print( f"Writing config file: {cfg_file}")
#                if os.path.exists(cfg_file): os.remove(cfg_file)
                conf_txt = mode_conf_txt if mode == "configuration" else '\n'.join( [ mode_conf_txt, globals ] )
                cfile_handle.write( conf_txt )

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
        plot = ipw.VBox([ collapsibles, ufm().gui() ], layout=ipw.Layout( flex='1 0 700px' ), border=css_border )
        control = ipw.VBox( [ am().gui(), mm().gui(), gm().gui() ], layout=ipw.Layout( flex='0 0 700px'), border=css_border )
        gui = ipw.HBox( [control, plot ], layout=ipw.Layout( width='100%' ) )
        self.save_config()
        if embed: ActionsManager.instance().embed()
        return gui

    def refresh_all(self):
        self.save_config()
        for config_class in self.config_classes: config_class.instance().refresh()
        print( "Refreshed Application")

    def __delete__(self, instance):
        self.save_config()

    def show_gpu_usage(self):
        os.system("nvidia-smi")





