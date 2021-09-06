from typing import List, Union, Tuple, Optional, Dict, Callable
from matplotlib.figure import Figure
from functools import partial
from collections import OrderedDict
import ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm, exception_handled
import traitlets as tl

class LayerPanel:

    def __init__(self, figure: Figure, name: str, ival: float, active: bool, handle_alpha_change: Callable ):
        self.name = name
        self.figure: Figure = figure
        self._slider = ipw.FloatSlider( ival, min=0.0, max=1.0, description=name )
        self._checkbox = ipw.Checkbox( active, description='')
        self._slider.observe( handle_alpha_change, names='value', type='change')
        self._checkbox.observe(handle_alpha_change, names='value', type='change')

    def gui(self) -> ipw.Box:
        buttonBox = ipw.HBox( [ self._slider, self._checkbox ], layout = ipw.Layout( width = "100%" ) )
        return buttonBox

class LayersManager(object):

    def __init__(self, figure: Figure ):
        super(LayersManager, self).__init__()
        self.figure: Figure = figure
        self._wGui: ipw.Box = None
        self._layers: OrderedDict[str,LayerPanel] = OrderedDict()

    def gui( self ) -> ipw.Box:
        if self._wGui is None:
            self._wGui = self._createGui( )
        return self._wGui

    def add_layer( self, name, ival: float, active: bool, handle_alpha_change: Callable ):
        self._layers[name] = LayerPanel( self.figure, name, ival, active, handle_alpha_change )

    def layer( self, name: str ) -> Optional[LayerPanel]:
        return self._layers.get( name, None )

    def _createGui( self ) -> ipw.Box:
        buttonBox =  ipw.VBox( [ layer.gui() for layer in self._layers.values() ] )
        buttonBox.layout = ipw.Layout( width = "100%" )
        return buttonBox

