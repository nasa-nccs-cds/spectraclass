import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable
import time
from functools import partial
import xarray as xa
import numpy as np
import ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.model.base import SCSingletonConfigurable

def am() -> "ActionsManager":
    return ActionsManager.instance()

class ActionsManager(SCSingletonConfigurable):

    def __init__(self):
        super(ActionsManager, self).__init__()
        self._wGui: ipw.Box = None
        self._buttons = {}

    def gui(self, **kwargs ) -> ipw.Box:
        if self._wGui is None:
            self._wGui = self._createGui( **kwargs )
        return self._wGui

    def on_button_click( self, task, button: ipw.Button = None ):
        from spectraclass.data.base import DataManager
        DataManager.instance().execute_task( task )

    def _createGui( self, **kwargs ) -> ipw.Box:
        from spectraclass.model.labels import LabelsManager
        for task in [ "embed", "mark", "spread", "distance", "undo", "clear" ]:
            button = ipw.Button( description=task, border= '1px solid gray' )
            button.layout = ipw.Layout( width='auto', flex="1 0 auto" )
            button.on_click( partial( self.on_button_click, task ) )
            self._buttons[ task ] = button
        buttonBox =  ipw.HBox( list(self._buttons.values()) )
        buttonBox.layout = ipw.Layout( width = "100%" )
        classes: ipw.DOMWidget = LabelsManager.instance().gui()
        gui = ipw.VBox([ classes, buttonBox ], layout = ipw.Layout( width="100%", justify_content="space-between", flex='0 0 70px', border= '2px solid firebrick' )  )
        return gui

def cm() -> "ControlsManager":
    return ControlsManager.instance()

class ControlsManager(SCSingletonConfigurable):

    def __init__(self):
        super(ControlsManager, self).__init__()
        self._wGui: ipw.Tab = None
        self._buttons = {}

    def gui(self, **kwargs ) -> ipw.Box:
        if self._wGui is None:
            self._wGui = self._createGui( **kwargs )
        return self._wGui

    def _createGui( self, **kwargs ) -> ipw.Box:
        wTab = ipw.Tab()
        tabNames = [ "data", "pointcloud", "map", "graph", "google" ]
        children = []
        for iT, title in enumerate( tabNames ):
            wTab.set_title( iT, title )
            children.append( self.createPanel(title) )
        wTab.children = children
        return wTab

    def createPanel(self, title: str ):
        return ipw.VBox()


def ufm() -> "UserFeedbackManager":
    return UserFeedbackManager.instance()

class UserFeedbackManager(SCSingletonConfigurable):

        def __init__(self):
            super(UserFeedbackManager, self).__init__()
            self._wGui: ipw.HTML = ipw.HTML( value='', placeholder='', description='messages:',
                                             layout = ip.Layout(width="100%"), border= '1px solid dimgrey' )

        def gui(self, **kwargs) -> ipw.HTML:
            return self._wGui

        def show(self, message: str, color: str = "white" ):
            self._wGui.value = f'<p style="color:{color}"><b>{message}</b></p>'

        def clear(self):
            self._wGui.value = ""



