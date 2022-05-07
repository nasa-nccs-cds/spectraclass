import os, ipywidgets as ipw
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.gui.control import UserFeedbackManager, ufm
import numpy as np
import xarray as xa

def em():
    rv = EventManager()
    return rv

class ActionEvent(object):

    def __init__( self, type: str ):
        super(ActionEvent, self).__init__()
        self._type = type

class LabelEvent(ActionEvent):

    def __init__( self, type: str, label_map: np.ndarray ):
        super(LabelEvent, self).__init__( type )
        self._label_map = label_map

    @property
    def label_map(self):
        return self._label_map

class EventManager(SCSingletonConfigurable):

    def __init__(self):
        super(EventManager, self).__init__()
        self._action_events = []
        self._event_listeners = []

    def addActionEvent(self, event: ActionEvent ):
        self._action_events.append( event )

    def popActionEvent(self) -> ActionEvent:
        return self._action_events.pop()

    def lastActionEvent(self) -> ActionEvent:
        return self._action_events[-1]