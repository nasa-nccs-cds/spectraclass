import hvplot.xarray
import holoviews as hv
import panel as pn
from scipy.io import loadmat
import os, numpy as np
import xarray as xa

data_dir = "/Users/tpmaxwel/GDrive/Tom/Data/Aviris/Salinas/"
gt_file = os.path.join( data_dir, "Salinas_gt.mat" )
hs_file = os.path.join( data_dir, "Salinas_corrected.mat" )
gt_varname = 'salinas_gt'
hs_varname = 'salinas_corrected'

def readMatArray( filename: str, varname: str ) -> xa.DataArray:
    gtdset = loadmat( filename )
    gtarray: np.ndarray = gtdset[varname]
    alldims = [ 'y', 'x', 'band' ]
    print( f"Reading variable '{varname}' from Matlab dataset '{filename}': {gtdset['__header__']}")
    dims = alldims[:gtarray.ndim]
    coords = { dims[i]: np.array( range(gtarray.shape[i]) ) for i in range(gtarray.ndim) }
    return xa.DataArray( gtarray, coords, dims, varname )

gt_array: xa.DataArray = readMatArray( gt_file, gt_varname )
print( f"shape = {gt_array.shape}, max = {gt_array.values.max()}, min = {gt_array.values.min()}, ")

hs_array: xa.DataArray = readMatArray( hs_file, hs_varname )
print( f"shape = {hs_array.shape}, max = {hs_array.values.max()}, min = {hs_array.values.min()}, ")

iBand = 10
gt_plot = gt_array.hvplot.image(cmap='Category20')
hs_plot = hs_array[:,:,iBand].hvplot.image(cmap='jet')
pn.Row( hs_plot, gt_plot ).show("Salinas")