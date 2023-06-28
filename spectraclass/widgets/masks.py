import holoviews as hv, panel as pn
from holoviews import opts, streams
from copy import deepcopy
from panel.layout import Panel
from spectraclass.util.logs import lgm, exception_handled, log_timing
from typing import List, Union, Tuple, Optional, Dict
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.application.controller import app
import numpy as np
from panel.widgets import Button, Select

def mm() -> "MaskManager":
    return MaskManager.instance()

class MaskManager(SCSingletonConfigurable):

    def __init__(self ):
        super(MaskManager, self).__init__()
        self.block_selection = pn.widgets.Toggle(name='Select Tiles', button_type='primary')
        self.block_selection_watcher = self.block_selection.param.watch( self.on_block_selection, ['value'], onlychanged=False)

    def on_block_selection(self, event ):
        print( f" on_block_selection: {event}")

    def get_control_panel(self):
        return pn.Column([self.block_selection])