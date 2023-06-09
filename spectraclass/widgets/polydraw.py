import holoviews as hv
import panel as pn
import copy
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
from holoviews import opts, streams
from panel.widgets import Button

hv.extension('bokeh')

class RegionSelector:

    def __init__(self, polys=None ):
        self.poly = hv.Polygons( [] if polys is None else polys )
        self.poly_stream = streams.PolyDraw(source=self.poly, drag=False, show_vertices=True, num_objects=1, styles={ 'fill_color': [ 'green' ] })
        self.select_button: Button = Button( name='Select', button_type='primary')
        self.buttonbox = pn.Row( self.select_button )
        self.selection = self.poly.opts(opts.Polygons(fill_alpha=0.3, active_tools=['poly_draw']))
        self.selected = hv.DynamicMap( self.get_selection, streams=[ self.select_button.param.clicks ] )
        self.selected_regions = []

    def get_poly_data1(self):
        pdata: Dict = self.poly_stream.data
        if pdata is not None:
            for k,v in pdata.items():
                if   k == "xs": result['x'] = v[0]
                elif k == "ys": result['y'] = v[0]
            result[ 'class' ] = 0
        return result

    def get_poly_data(self) -> Dict:
        polys = self.poly_stream.element.split(datatype='dictionary')
        pdata = polys[0] if len(polys) else None
        print( f" poly(split)= {pdata}")
        return pdata

    def get_selection(self, *args, **kwargs ):
        pdata = self.get_poly_data()
        if pdata is not None: self.selected_regions.append( pdata )
        print(f" get_selection.selected_regions = {self.selected_regions}")
        return hv.Polygons( self.selected_regions ).opts( opts.Polygons(fill_alpha=0.6) )

    def panel(self):
        return pn.Column( self.selection+self.selected, self.buttonbox )