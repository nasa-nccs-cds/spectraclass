import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable
import time
import pandas as pd
from functools import partial
import xarray as xa
import numpy as np
import holoviews as hv
import ipywidgets as ipw
import panel as pn
from panel.pane import Alert
import traitlets.config as tlc
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import traitlets as tl

def get_parameter_table( parms: Dict, **opts) -> hv.Table:
    parms.pop("")
    lgm().log( f"****>> get_parameter_table, parms={parms}")
    df = pd.DataFrame( parms )
    return hv.Table(df).options( selectable=True, editable=False, **opts)

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
        elif task == "mask":
            app().mask()
        elif task == "distance":
            app().display_distance()

    def _createGui( self, **kwargs ) -> ipw.Box:
        from spectraclass.model.labels import LabelsManager
        for task in [ "embed", "distance", "propagate", "undo", "clear"]:
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

    @exception_handled
    def _createGui( self, **kwargs ) -> ipw.Box:
        wTab = ipw.Tab()
        tabNames = [  "layers", "selection", "learning", "threshold", "persist", "cluster" ]
        children = []
        for iT, title in enumerate( tabNames ):
            children.append( self.createPanel(title) )
        wTab.children = children
        for iT, title in enumerate( tabNames ):
            wTab.set_title( iT, title )
        return wTab

    def createPanel(self, title: str ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.data.base import DataManager, dm
        from spectraclass.learn.cluster.manager import clm
        from spectraclass.learn.manager import ClassificationManager, cm
        if title   == "layers":     return  mm().layers.gui()
        elif title == "persist":    return  cm().create_persistence_gui()
        elif title == "selection":  return  mm().get_selection_panel()
        elif title == "learning":   return  cm().gui()
        elif title == "cluster":    return  clm().gui()
        elif title == "threshold":  return  mm().get_threshold_panel()
        elif title == "reduction":  return  dm().modal.getCreationPanel()
        elif title == "embedding":  return  dm().modal.getConfigPanel()
        else: return ipw.VBox([])

def ufm() -> "UserFeedbackManager":
    return UserFeedbackManager.instance()

class UserFeedbackManager(SCSingletonConfigurable):

        def __init__(self):
            super(UserFeedbackManager, self).__init__()
            self._wGui: Alert = Alert()

        def gui(self) -> Alert:
            return self._wGui

        def show(self, message: str, alert_type: str = "info", **kwargs ):  #  alert_types: primary, secondary, success, danger, warning, info, light, dark.
            self._wGui.alert_type = alert_type
            self._wGui.object = message
            lgm().log( message )

        def clear(self):
            self._wGui.object = ""



