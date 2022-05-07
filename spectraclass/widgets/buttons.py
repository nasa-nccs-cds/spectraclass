from typing import List, Union, Tuple, Optional, Dict, Callable
from IPython.core.debugger import set_trace
from functools import partial
import ipywidgets as widgets

class ToggleButton:

    def __init__(self, icons: List[str], states: List[str], tooltips: List[str], **kwargs ):
        self._states = states
        self._icons = icons
        self._tooltips = tooltips
        button_width = kwargs.get( 'width', "35px" )
        self._nStates = len(icons)
        self._button_layout = widgets.Layout(width=button_width, max_width=button_width, min_width=button_width)
        assert len( states) == self._nStates, "icons and states must have the same length"
        assert len( tooltips ) == self._nStates, "icons and tool_tips must have the same length"
        self._gui = None
        self._state_index = 0
        self._listeners: List[Callable[[str],None] ] = []

    @property
    def state(self):
        return self._states[ self._state_index ]

    @property
    def icon(self):
        return self._icons[ self._state_index ]

    @property
    def tooltip(self):
        return self._tooltips[self._state_index]

    def add_listener(self, listener: Callable[[str],None]  ):
        self._listeners.append( listener )

    def _on_button_click( self, instance: widgets.Button ):
        self._state_index = ( self._state_index + 1 ) % self._nStates
        instance.icon = self.icon
        instance.tooltip = self.tooltip
        for listener in self._listeners:
            listener( self.state )

    def gui(self):
        if self._gui == None:
            self._gui = widgets.Button( disabled=False, icon=self._icons[0], tooltip=self._tooltips[0], layout=self._button_layout, border= '1px solid dimgrey'  )
            self._gui.on_click( self._on_button_click)
        return self._gui
