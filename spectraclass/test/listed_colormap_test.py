import xarray as xa
import numpy as np
import matplotlib as mpl
from matplotlib.image import AxesImage
from typing import List, Union, Tuple, Optional, Dict
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.tile.tile import Block
import random, matplotlib.pyplot as plt
colorstretch = 1.0

def scale_to_bounds( raster: xa.DataArray, bounds: Tuple[float, float]) -> xa.DataArray:
    vmin = raster.min(dim=raster.dims[:2], skipna=True)
    vmax = raster.max(dim=raster.dims[:2], skipna=True)
    scale = (bounds[1] - bounds[0]) / (vmax - vmin)
    return (raster - vmin) * scale + bounds[0]

def get_color_bounds( color_values: List[float] ) -> List[float]:
    color_bounds = []
    for iC, cval in enumerate( color_values ):
        if iC == 0: color_bounds.append( cval - 0.5 )
        else: color_bounds.append( (cval + color_values[iC-1])/2.0 )
    color_bounds.append( color_values[-1] + 0.5 )
    return color_bounds

def plotRaster( raster: xa.DataArray, **kwargs):
    from matplotlib.colorbar import Colorbar
    from spectraclass.application.controller import app
    ax = kwargs.pop('ax', None)
    showplot = (ax is None)
    if showplot: fig, ax = plt.subplots(1, 1)
    colors = kwargs.pop('colors', None)
    title = kwargs.pop('title', raster.name)
    rescale = kwargs.pop('rescale', None)
    colorbar = kwargs.pop('colorbar', True)
    x = raster.coords[raster.dims[1]].values
    y = raster.coords[raster.dims[0]].values
    try:
        xstep = (x[1] - x[0]) / 2.0
    except IndexError:
        xstep = .1
    try:
        ystep = (y[1] - y[0]) / 2.0
    except IndexError:
        ystep = .1
    left, right = x[0] - xstep, x[-1] + xstep
    bottom, top = y[-1] + ystep, y[0] - ystep
    defaults = dict(origin='upper', interpolation='nearest')
    defaults["alpha"] = kwargs.get("alpha", 1.0)
    cbar_kwargs = {}
    if colors is None:
        defaults.update(dict(cmap=app().color_map))
    else:
        rgbs = [cval[2] for cval in colors]
        cmap: ListedColormap = ListedColormap(rgbs)
        color_values = [ float(cval[0]) for cval in colors ]
        color_bounds = get_color_bounds(color_values)
        norm = mpl.colors.BoundaryNorm(color_bounds, len(colors))
        cbar_kwargs.update( dict(cmap=cmap, norm=norm, boundaries=color_bounds, ticks=color_values, spacing='proportional') )
        defaults.update(dict(cmap=cmap, norm=norm))
    if not hasattr(ax, 'projection'): defaults['aspect'] = 'auto'
    vrange = kwargs.pop('vrange', None)
    if vrange is not None:
        defaults['vmin'] = vrange[0]
        defaults['vmax'] = vrange[1]
    defaults.update(kwargs)
    if defaults['origin'] == 'upper':
        defaults['extent'] = [left, right, bottom, top]
    else:
        defaults['extent'] = [left, right, top, bottom]
    if rescale is not None:
        raster = scale_to_bounds(raster, rescale)
    print(f"\n $$$COLOR: Plotting tile image with parameters: {defaults}\n")
    img = ax.imshow(raster.data, **defaults)
    ax.set_title(title)
    if colorbar and (raster.ndim == 2):
        cbar: Colorbar = ax.figure.colorbar(img, ax=ax, **cbar_kwargs)
        if colors is not None:
            cbar.set_ticklabels([cval[1] for cval in colors])
    if showplot: plt.show()
    return img

def plotOverlayImage( raster: np.ndarray, ax, colors, extent: Tuple[float] = None ):
    defaults = dict(origin='upper', alpha=0.0, cmap = ListedColormap(colors) )
    if extent is not None: defaults['extent'] = extent
    if not hasattr(ax, 'projection'): defaults['aspect'] = 'auto'
    print( f"\n $$$COLOR: Plotting overlay image with parameters: {defaults}, data range = {[raster.min(), raster.max()]}, raster type = {raster.dtype}, shape = {raster.shape}\n")
    img = ax.imshow(raster, **defaults)
    return img

extent = None
color_band = 125
alpha = 0.5

colors = [ "yellow", "cyan", "green", "magenta", "blue", "red" ]
values = range( len(colors) )
lables = [ str(v) for v in values ]
cvals = list(zip( values,lables,colors ))

dm: DataManager = DataManager.initialize("demo3",'aviris')
project_data = dm.loadCurrentProject("main")
block: Block = tm().getBlock()
band_data: xa.DataArray = block.data[ color_band ]
overlay_data = np.random.randint( 0, 6, band_data.shape )
init_overlay_data = np.zeros( band_data.shape, np.int )

fig, ax = plt.subplots(1,1)
base_image: AxesImage = plotRaster( band_data, ax = ax )

overlay_image: AxesImage = plotRaster( band_data.copy(data=init_overlay_data), alpha=0.0, ax=ax, colors=cvals )
overlay_image.set_alpha( alpha )
overlay_image.set_data( overlay_data )

plt.show()