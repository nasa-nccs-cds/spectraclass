import holoviews as hv, panel as pn
from holoviews import opts, streams
from copy import deepcopy
from typing import List, Union, Tuple, Optional, Dict
import numpy as np
from panel.widgets import Button, Select
class_colors = [ 'blue', 'yellow', 'green', 'cyan', 'brown', 'magenta' ]

def centers( polygons: hv.Polygons ) -> str:
    pds = []
    for pd in polygons.data:
        x: np.ndarray = pd['x']
        y: np.ndarray = pd['y']
        pds.append( f"({x.mean():0.2f},{y.mean():0.2f})" )
    return str(pds)

class RegionSelector:

    def __init__(self ):
        self._addclks = 0
        self._removeclks = 0
        self.poly = hv.Polygons([])
        self.poly_stream = streams.PolyDraw(source=self.poly, drag=False, num_objects=1, show_vertices=True, styles={'fill_color': ['red']})
        self.select_button: Button = Button( name='Select', button_type='primary')
        self.undo_button: Button = Button( name='Undo', button_type='warning')
        self.indicator = streams.SingleTap(transient=True)
        self.selections = []

    def get_selections( self, addclicks: int, removeclicks: int ):
      if addclicks > self._addclks:
        cindex = addclicks % len(class_colors)
        ccolor = class_colors[cindex]
        selection: hv.Polygons = self.poly_stream.element.opts( color=ccolor, line_width=1, alpha=0.3, line_color="black" )
        print( f"Add poly_stream element: {centers(selection)}")
        self.selections.append( hv.Polygons( deepcopy(selection.data) ) )
      if removeclicks > self._removeclks:
        removed = self.selections.pop()
        print(f"Remove selected element: {centers(removed)}")
      self._addclks, self._removeclks = addclicks, removeclicks
      print(f"Current Selections: {[ centers(s) for s in self.selections ]}")
      return hv.Overlay( self.selections )

    def indicate( self, x, y ):
      print( f"indicate: {x} {y}")
      return hv.Points( [(x,y)] ).opts( color="black" )

    def panel(self):
        indication = hv.DynamicMap( self.indicate, streams=[self.indicator])
        selected = hv.DynamicMap( self.get_selections, streams=dict( addclicks=self.select_button.param.clicks, removeclicks=self.undo_button.param.clicks ) )

        canvas = self.poly.opts( opts.Polygons(fill_alpha=0.3, active_tools=['poly_draw']))
        buttonbox = pn.Row( self.select_button, self.undo_button )
        return pn.Column( canvas*selected*indication,buttonbox )