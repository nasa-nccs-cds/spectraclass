import numpy as np
from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.layouts import gridplot
from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass

app = Spectraclass.instance()
dm: DataManager = app.configure("demo1",'aviris')
dataset = dm.loadCurrentProject( "main" )

input: np.ndarray = dataset['raw'].values
x: np.ndarray = np.arange( input.shape[0] )
r: np.ndarray = dataset['reproduction'].values

# nplots = 11
# dp = math.floor(255/(nplots-1))
# p = figure( plot_width=1600, plot_height=800 )
# xcoords =  [ x for i in range(nplots) ]
# ycoords =  [ y[i] for i in range(nplots) ]
# rcoords =  [ r[i] for i in range(nplots) ]
# source = ColumnDataSource(data=dict( xs=xcoords, ys=rcoords, colors256=[i*dp for i in range(nplots)] ))
# p.multi_line('xs', 'ys', source=source, color=linear_cmap('colors256', "Turbo256", 0, 255))
# show(p)

def rescale( x ): return x/x.mean()


nrows, ncols = 3,4
pw, ph = 1800//ncols, 800//nrows
plots = []
for ir in range(nrows):
    rows = []
    for ic in range(ncols):
        ip = ic + ir*ncols
        p = figure( plot_width=pw, plot_height=ph )
        xcoords =  [ x, x ]
        ycoords =  [ rescale(input[ip]), rescale(r[ip]) ]
        source = ColumnDataSource(data=dict( xs=xcoords, ys=ycoords, colors256=[20,240] ))
        p.multi_line('xs', 'ys', source=source, color=linear_cmap('colors256', "Turbo256", 0, 255))
        rows.append(p)
    plots.append(rows)

grid = gridplot(plots)
show(grid)