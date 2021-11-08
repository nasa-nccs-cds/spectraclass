import holoviews as hv
import param
from holoviews import opts, streams
from collections import OrderedDict
from holoviews.plotting.links import DataLink

class PointColorOp(hv.Operation):
    color = param.String( default="blue" )

    def _process( self, element, key=None ):
        xs = element.dimension_values(0)
        ys = element.dimension_values(1)
        colors = element.dimension_values(2)
        new_colors = [ self.color if c == "white" else c for c in colors ]
        element = element.clone( (xs, ys, new_colors) )
        return element.opts( opts.Points( color='color' ) )

class PointSelection:

    def __init__(self, **kwargs ):
        self.points = hv.Points( ([], [], []), vdims='color')
        self.point_size = kwargs.get( 'size', 5 )
        self.color_op = PointColorOp()
        self.point_stream = streams.PointDraw( source=self.points, empty_value='white' )
        self.table = hv.Table(self.points, ['x', 'y'], 'color')
        DataLink(self.points, self.table)

    def set_color(self, color: str ):
        self.point_stream.empty_value = color

    def plot(self):
        return hv.DynamicMap( PointColorOp(self.points), streams=[self.points] ).opts( opts.Points( active_tools=['point_draw'], size=self.point_size ) )



