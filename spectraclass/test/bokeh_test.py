from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

output_file("cmap.html")

p = figure()

source = ColumnDataSource(data=dict(
    xs=[[0,1] for i in range(256)],     # x coords for each line (list of lists)
    ys=[[i, i+10] for i in range(256)], # y coords for each line (list of lists)
    foo=list(range(256))                # data to use for colormapping
))

p.multi_line('xs', 'ys', source=source,
             color=linear_cmap('foo', "Viridis256", 0, 255))

show(p)