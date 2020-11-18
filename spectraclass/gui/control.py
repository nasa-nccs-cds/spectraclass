import ipywidgets as ip
from typing import List, Union, Tuple, Optional, Dict, Callable
import time
from functools import partial
import xarray as xa
import numpy as np
import ipywidgets as ipw
from .points import PointCloudManager
import traitlets.config as tlc
from spectraclass.model.base import AstroConfigurable

class ActionsPanel(tlc.SingletonConfigurable, AstroConfigurable):

    def __init__(self):
        super(ActionsPanel, self).__init__()
        self._wGui: ipw.Box = None
        self._buttons = {}

    def gui(self, **kwargs ) -> ipw.Box:
        if self._wGui is None:
            self._wGui = self._createGui( **kwargs )
        return self._wGui

    def on_button_click( self, task, button: ipw.Button = None ):
        from .table import TableManager
        tmgr = TableManager.instance()
        if task ==   "embed":  PointCloudManager.instance().reembed()
        elif task == "mark":   tmgr.mark_selection()
        elif task == "spread": tmgr.spread_selection()
        elif task == "clear":  tmgr.clear_current_class()
        elif task == "undo":   tmgr.undo_action()
        elif task == "distance": tmgr.display_distance()

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
        gui = ipw.VBox([ classes, buttonBox ], layout = ipw.Layout( width="100%", justify_content="space-between", flex='0 0 90px', border= '2px solid firebrick' )  )
        return gui

    def embed(self):
        self.on_button_click("embed")

