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

    def get_control_panel(self):
        return pn.Column([])