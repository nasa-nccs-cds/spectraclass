import holoviews as hv, panel as pn
from holoviews import opts, streams
from panel.widgets import Button, Select
class_colors = [ 'blue', 'yellow', 'green', 'cyan', 'brown', 'magenta' ]

class RegionSelector:

    def __init__(self ):
        self._addclks = 0
        self._removeclks = 0
        self.poly = hv.Polygons([])
        self.poly_stream = streams.PolyDraw(source=self.poly, drag=False, num_objects=1, show_vertices=True, styles={'fill_color': ['red']})
        self.select_button: Button = Button( name='Select', button_type='primary')
        self.undo_button: Button = Button( name='Undo', button_type='primary')
        self.indicator = streams.SingleTap(transient=True)
        self.selections = []

    def get_selections( self, addclicks: int, removeclicks: int ):
      if addclicks > self._addclks:
        cindex = addclicks % len(class_colors)
        ccolor = class_colors[cindex]
        selection = self.poly_stream.element.opts( color=ccolor, line_width=1, alpha=0.3, line_color="black" )
        self.selections.append( selection )
      if removeclicks > self._removeclks:
        self.selections.pop()
      self._addclks, self._removeclks = addclicks, removeclicks
      return hv.Overlay( self.selections )

    def indicate( self, x, y ):
      print( f"indicate: {x} {y}")
      return hv.Points( [(x,y)] ).opts( color="black" )

    def panel(self):
        indication = hv.DynamicMap( self.indicate, streams=[self.indicator])
        selected = hv.DynamicMap( self.get_selections, streams=dict( addclicks=self.select_button.param.clicks, removeclicks=self.undo_button.param.clicks ) )

        canvas = self.poly.opts( opts.Polygons(fill_alpha=0.3, active_tools=['poly_draw']))
        buttonbox = pn.Row( self.select_button, self.undo_button )
        pn.Column( canvas*selected*indication,buttonbox )