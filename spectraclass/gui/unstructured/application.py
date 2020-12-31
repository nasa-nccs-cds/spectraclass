import os, ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.model.base import SCConfigurable

class Spectraclass(tlc.SingletonConfigurable, SCConfigurable):

    HOME = os.path.dirname( os.path.dirname( os.path.dirname(os.path.realpath(__file__)) ) )
    custom_theme = False

    def __init__(self):
        super(Spectraclass, self).__init__()

    def configure( self, name: str ):
        from spectraclass.data.base import DataManager
        DataManager.instance().name = name
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

    def save_config( self ):
        from spectraclass.data.base import DataManager
        conf_dict = self.generate_config_file()
        globals = conf_dict.pop( 'global', {} )
        for mode, mode_conf_txt in conf_dict.items():
            cfg_file = os.path.realpath( DataManager.instance().config_file(mode) )
            os.makedirs( os.path.dirname(cfg_file), exist_ok=True )
            with open( cfg_file, "w" ) as cfile_handle:
#                if os.path.exists(cfg_file): os.remove(cfg_file)
                print( f"Writing config file: {cfg_file}")
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
        from spectraclass.gui.graph import GraphManager
        from spectraclass.gui.points import PointCloudManager
        from spectraclass.gui.unstructured.table import TableManager
        from spectraclass.gui.control import ActionsManager

        self.set_spectraclass_theme()
        self.configure("spectraclass")
        css_border = '1px solid blue'

        tableManager = TableManager.instance()
        graphManager = GraphManager.instance()
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





