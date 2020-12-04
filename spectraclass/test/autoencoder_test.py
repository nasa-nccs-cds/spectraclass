from spectraclass.data.manager import DataManager, ModeDataManager
from spectraclass.gui.application import Spectraclass
import numpy as np
import math
from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.layouts import gridplot

app = Spectraclass.instance()
app.configure("spectraclass")

dm: DataManager = DataManager.instance()
mdm: ModeDataManager = dm.mode_data_manager

mdm.model_dims = 32
mdm.subsample = 100
mdm.reduce_method = "autoencoder"
mdm.reduce_nepochs = 2000

dataset = mdm.prepare_inputs( write=False )

print( f"prepare_inputs: {dataset}" )

input: np.ndarray = dataset['embedding'].values
x: np.ndarray = dataset['plot-x'].values
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


nrows, ncols = 3,4
pw, ph = 1800//ncols, 800//nrows
plots = []
for ir in range(nrows):
    rows = []
    for ic in range(ncols):
        ip = ic + ir*ncols
        p = figure( plot_width=pw, plot_height=ph )
        xcoords =  [ x, x ]
        ycoords =  [ input[ip], r[ip] ]
        source = ColumnDataSource(data=dict( xs=xcoords, ys=ycoords, colors256=[20,240] ))
        p.multi_line('xs', 'ys', source=source, color=linear_cmap('colors256', "Turbo256", 0, 255))
        rows.append(p)
    plots.append(rows)

grid = gridplot(plots)
show(grid)