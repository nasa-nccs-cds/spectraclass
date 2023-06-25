import holoviews as hv, panel as pn
from holoviews import opts, streams
from copy import deepcopy
from panel.layout import Panel
from spectraclass.util.logs import lgm, exception_handled, log_timing
from typing import List, Union, Tuple, Optional, Dict
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.gui.spatial.widgets.markers import Marker
from spectraclass.application.controller import app
import numpy as np
from panel.widgets import Button, Select

def rs() -> "RegionSelector":
    return RegionSelector.instance()

def centers( polygons: hv.Polygons ) -> str:
    pds = []
    for pd in polygons.data:
        x: np.ndarray = pd['x']
        y: np.ndarray = pd['y']
        pds.append( f"({x.mean():0.2f},{y.mean():0.2f})" )
    return str(pds)

class RegionSelector(SCSingletonConfigurable):

    def __init__(self ):
        super(RegionSelector, self).__init__()
        self._addclks = 0
        self._removeclks = 0
        self.poly = hv.Polygons([])
        self.selected_regions = hv.Polygons([])
        self.poly_stream = streams.PolyDraw(source=self.poly, drag=False, num_objects=1, show_vertices=True, styles={'fill_color': ['red']})
        self.poly_edit = streams.PolyEdit(source=self.selected_regions, vertex_style={'color': 'red'})
        self.select_button: Button = Button( name='Select', button_type='primary')
        self.learn_button: Button = Button( name='Learn', button_type='primary')
        self.selections = []
        self.selected  = hv.DynamicMap( self.get_selections, streams=dict( clicks=self.select_button.param.clicks ) )
        self.canvas = self.poly.opts( opts.Polygons(fill_alpha=0.3, active_tools=['poly_draw','poly_edit']))
        self.buttonbox = pn.Row( self.select_button, self.learn_button )
        self.markers: Dict[ hv.Polygons, Marker ] = {}
        self.learn_button.on_click( self.learn_classification )

    @exception_handled
    def reset(self, *args, **kwargs ):
        self.selected.reset()

    @exception_handled
    def learn_classification(self, *args, **kwargs ):
        from spectraclass.reduction.vae.trainer import mt
        mt().train()

    @exception_handled
    def get_selections( self, clicks: int ):
      from spectraclass.data.spatial.tile.manager import TileManager, tm
      if clicks > self._addclks:
        ic, ccolor = lm().selectedColor( True )
        ufm().show( f"Selecting region as class '{lm().selectedLabel}({ic}): color={ccolor}'")
        selection: hv.Polygons = self.poly_stream.element
        print( f"Add poly_stream element: {centers(selection)}-> ic={ic}, color={ccolor}")
        print( f"PolyData: {selection.data}")
        spoly = hv.Polygons( selection.data ).opts( fill_color=ccolor, line_width=1, alpha=0.3, line_color="black" )

      #  self.selected_regions

        self.selections.append( spoly )
        marker = tm().get_region_marker( spoly.data[0] )
        self.markers[ spoly ] = marker
#        app().add_marker(marker)
      self._addclks = clicks
      print(f" *** Current Selections: {[ centers(s) for s in self.selections ]}")
      return hv.Overlay( self.selections )

    @exception_handled
    def indicate( self, x, y ):
      print( f"indicate: {x} {y}")
      return hv.Points( [(x,y)] ).opts( color="black" )

    def panel(self):
        return pn.Column( self.canvas*self.selected, self.buttonbox )

    def get_control_panel(self) -> Panel:
        return self.buttonbox

    def get_selector(self):
        return self.canvas * self.selected