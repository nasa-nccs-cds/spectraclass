from __future__ import annotations
from typing import List, Union, Tuple, Optional, Dict, Callable
from collections import OrderedDict
import ipywidgets as ipw
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import traitlets as tl

class Layer(tlc.Configurable):
    alpha = tl.Float(1.0).tag(config=True, sync=True)
    visible = tl.Bool(True).tag(config=True, sync=True)

    def __init__(self, name: str, alpha: float, visible: bool, callback: Callable[[Layer], None]):
        super(Layer, self).__init__()
        self.name = name
        self.alpha = alpha
        self.callback = callback
        self.visible = visible
        self._notify = True
        self._label = ipw.Label( value=name, layout = ipw.Layout( width = "100px" ) )
        self._slider = self.getSlider()
        self._checkbox = self.getCheckBox()
        self.observe( self.on_alpha_change, names=["alpha","visible"] )

    def on_alpha_change( self, *args ):
 #       lgm().log( f' ** on_alpha_change[{self.name}]: alpha={self.alpha}, visible={self.visible}, args={args}' )
        if self._notify:
            self.callback( self )

    def trigger( self, eps = 0.001 ):
        if self.alpha > (1.0 - eps):    self.alpha = self.alpha - eps
        else:                           self.alpha = self.alpha + eps

    @property
    def visibility(self) -> float:
        if self.visible: return self.alpha
        return 0.0

    def getSlider(self):
        slider = ipw.FloatSlider( self.alpha, description='alpha', min=0.0, max=1.0 )
        tl.link( (slider, "value"), (self, 'alpha') )
        return slider

    def getCheckBox( self ):
        checkbox = ipw.Checkbox( self.visible, description='visible' )
        tl.link((checkbox, "value"), (self, 'visible') )
        return checkbox

    def update(self, alpha: float, enabled: bool, **kwargs ):
        self._notify = kwargs.get('notify',False)
        self._slider.value = alpha
        self._checkbox.value = enabled
        self._notify = True

    def increment( self, increase: bool ):
        if increase:   self.alpha = min( 1.0, self.alpha + 0.1 )
        else:          self.alpha = max( 0.0, self.alpha - 0.1 )

    def gui(self) -> ipw.Box:
        return ipw.HBox( [self._label, self._slider, self._checkbox ] ) # , layout = ipw.Layout( width = "150px" ) )

class LayersManager(object):

    def __init__(self, callback: Callable[[Layer], None]):
        super(LayersManager, self).__init__()
        self.callback: Callable[[Layer], None] = callback
        self._layers: OrderedDict[str, Layer] = OrderedDict()

    def gui( self ) -> ipw.Box:
        return ipw.VBox( [ layer.gui() for layer in self._layers.values() ], layout = ipw.Layout( width = "100%" ) )

    def set_visibility(self, lname: str, alpha: float, enabled: bool, **kwargs ):
        self._layers[lname].update( alpha, enabled, **kwargs )

    def alpha(self, lname: str) -> float:
        return self._layers[lname].visibility

    def add( self, name: str, alpha: float, visible: bool ):
        self._layers[name] = Layer(name, alpha, visible, self.callback)

    def __call__(self, name: str ) -> Optional[Layer]:
        return self._layers.get( name, None )

