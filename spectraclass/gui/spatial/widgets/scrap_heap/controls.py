import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable
import time
from functools import partial
import xarray as xa
from collections import OrderedDict
import ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled
import traitlets as tl

def am() -> "ActionsManager":
    return ActionsManager.instance()

class ActionsManager(SCSingletonConfigurable):

    def __init__(self):
        super(ActionsManager, self).__init__()
        self._wGui: ipw.Box = None
        self._buttons: OrderedDict[str,ipw.Button] = OrderedDict()
        self._actions: OrderedDict[str,Callable] = OrderedDict()

    def gui(self, **kwargs ) -> ipw.Box:
        if self._wGui is None:
            self._wGui = self._createGui( **kwargs )
        return self._wGui

    def add_action( self, name, executable: Callable ):
        self._actions[name] = executable

    @exception_handled
    def on_button_click( self, action: str, button: ipw.Button = None ):
        lgm().log(f" ***  EXECUTE: {action}" )
        executable = self._actions.get( action, None )
        if executable is None: ufm().show( f"Action {action} is not defined")
        else: executable()

    def _createGui( self, **kwargs ) -> ipw.Box:
        for task in self._actions.keys():
            button = ipw.Button( description=task, border= '1px solid gray' )
            button.layout = ipw.Layout( width='auto', flex="1 0 auto" )
            button.on_click( partial( self.on_button_click, task ) )
            self._buttons[ task ] = button
        buttonBox =  ipw.HBox( list(self._buttons.values()) )
        buttonBox.layout = ipw.Layout( width = "100%" )
        return buttonBox

def pm() -> "ParametersManager":
    return ParametersManager.instance()

class ParametersManager(SCSingletonConfigurable):

    def __init__(self):
        super(ParametersManager, self).__init__()
        self._wGui: ipw.Tab = None
        self._buttons = {}

    def gui(self, **kwargs ) -> ipw.Box:
        if self._wGui is None:
            self._wGui = self._createGui( **kwargs )
        return self._wGui

    def _createGui( self, **kwargs ) -> ipw.Box:
        wTab = ipw.Tab()
        tabNames = [ "reduction", "embedding", "classification" ]
        children = []
        for iT, title in enumerate( tabNames ):
            wTab.set_title( iT, title )
            children.append( self.createPanel(title) )
        wTab.children = children
        return wTab

    def createPanel(self, title: str ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.data.base import DataManager, dm
        widgets = []
        if title == "classification":
            widgets.append( self.getFloatSlider( "Overlay Opacity", (0.0,1.0), mm(), "overlay_alpha" ) )
        elif title == "reduction":
            widgets.append( dm().modal.getCreationPanel() )
        elif title == "embedding":
            widgets.append( dm().modal.getConfigPanel() )
        return  ipw.VBox(widgets)

    def getFloatSlider(self, label: str, range: Tuple[float,float], observer: tlc.Configurable, linked_trait: str ):
        slider = ipw.FloatSlider( getattr(observer,linked_trait), description=label, min=range[0], max=range[1] )
        tl.link((slider, "value"), (observer, linked_trait) )
        return slider


def ufm() -> "UserFeedbackManager":
    return UserFeedbackManager.instance()

class UserFeedbackManager(SCSingletonConfigurable):

        def __init__(self):
            super(UserFeedbackManager, self).__init__()
            self._wGui: ipw.HTML = ipw.HTML( value='', placeholder='', description='messages:',
                                             layout = ip.Layout(width="100%"), border= '1px solid dimgrey' )

        def gui(self, **kwargs) -> ipw.HTML:
            return self._wGui

        def show(self, message: str, color: str = "yellow" ):
            self._wGui.value = f'<p style="color:{color}"><b>{message}</b></p>'

        def clear(self):
            self._wGui.value = ""



