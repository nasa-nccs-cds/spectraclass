import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable
import time
from functools import partial
import xarray as xa
import numpy as np
import ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import traitlets as tl

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

    @exception_handled
    def on_button_click( self, task, button: ipw.Button = None ):
        from spectraclass.application.controller import app
        lgm().log(f" ***  APP EXECUTE: {task}" )
        if task == "embed":
            app().embed()
        elif task == "mark":
            app().mark()
        elif task == "propagate":
            app().propagate_selection()
        elif task == "clear":
            app().clear()
        elif task == "undo":
            app().undo_action()
        elif task == "learn":
            app().learn()
        elif task == "mask":
            app().mask()
        elif task == "apply":
            app().classify()
        elif task == "distance":
            app().display_distance()

    def _createGui( self, **kwargs ) -> ipw.Box:
        from spectraclass.model.labels import LabelsManager
#        for task in [ "embed", "mark", "spread", "distance", "learn", "classify", "mask", "undo", "clear" ]:
        for task in [ "embed", "propagate", "learn", "apply", "undo", "clear"]:
            button = ipw.Button( description=task, border= '1px solid gray' )
            button.layout = ipw.Layout( width='auto', flex="1 0 auto" )
            button.on_click( partial( self.on_button_click, task ) )
            self._buttons[ task ] = button
        buttonBox =  ipw.HBox( list(self._buttons.values()) )
        buttonBox.layout = ipw.Layout( width = "100%" )
        classes: ipw.DOMWidget = LabelsManager.instance().gui()
        gui = ipw.VBox([ classes, buttonBox ], layout = ipw.Layout( width="100%", justify_content="space-between", flex='0 0 70px', border= '2px solid firebrick' )  )
        return gui

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
        tabNames = [  "layers", "selection", "learning", "threshold"  ]
        children = []
        for iT, title in enumerate( tabNames ):
            wTab.set_title( iT, title )
            children.append( self.createPanel(title) )
        wTab.children = children
        return wTab

    def createPanel(self, title: str ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.learn.manager import ClassificationManager, cm
        widgets = []
        if title == "layers":
            widgets.append( mm().layers.gui() )
        if title == "selection":
            widgets.append(mm().get_selection_panel())
        if title == "learning":
            widgets.append( cm().gui() )
        if title == "threshold":
            widgets.append(mm().get_threshold_panel())
        elif title == "reduction":
            widgets.append( dm().modal.getCreationPanel() )
        elif title == "embedding":
            widgets.append( dm().modal.getConfigPanel() )
        return  ipw.VBox(widgets)

def ufm() -> "UserFeedbackManager":
    return UserFeedbackManager.instance()

class UserFeedbackManager(SCSingletonConfigurable):

        def __init__(self):
            super(UserFeedbackManager, self).__init__()
            self._wGui: ipw.HTML = ipw.HTML( value='', placeholder='', description='messages:',
                                             layout = ip.Layout(width="100%"), border= '1px solid dimgrey' )

        def gui(self, **kwargs) -> ipw.HTML:
            return self._wGui

        def show(self, message: str, color: str = "blue" ):
            self._wGui.value = f'<p style="color:{color}"><b>{message}</b></p>'

        def clear(self):
            self._wGui.value = ""



