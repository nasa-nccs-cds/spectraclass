import numpy as np
import xarray as xa
import pathlib
import traitlets as tl
import traitlets.config as tlc
import matplotlib as mpl
from typing import List, Union, Tuple, Optional, Dict
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from spectraclass.reduction.embedding import ReductionManager
from spectraclass.data.base import ModeDataManager
import matplotlib.pyplot as plt
from collections import OrderedDict
import os, math, pickle
import rioxarray as rio
from .modes import *


def dm():
    from spectraclass.data.base import DataManager
    return DataManager.instance()

def get_color_bounds( color_values: List[float] ) -> List[float]:
    color_bounds = []
    for iC, cval in enumerate( color_values ):
        if iC == 0: color_bounds.append( cval - 0.5 )
        else: color_bounds.append( (cval + color_values[iC-1])/2.0 )
    color_bounds.append( color_values[-1] + 0.5 )
    return color_bounds

# class MarkerManager:
#
#     def __init__(self, file_name: str, config: QSettings, **kwargs ):
#         self.file_name = file_name
#         self.names = None
#         self.colors = None
#         self.markers = None
#         self.config = config
#
#     @property
#     def file_path(self):
#         if self.file_name.startswith( "/" ):
#             return self.file_name
#         else:
#             data_dir = self.config.value( 'data/cache', "" )
#             return os.path.join( data_dir, self.file_name )
#
#     @property
#     def hasData(self):
#         return self.markers is not None
#
#     def writeMarkers(self, names, colors, markers ):
#         try:
#             with open( self.file_path, 'wb' ) as f:
#                 print( f"Saving {len(markers)} labeled points to file {self.file_path}")
#                 pickle.dump( [ names, colors, markers ], f )
#         except Exception as err:
#             print( f" Can't save markers: {err}")
#
#     def readMarkers(self):
#         try:
#             if os.path.isfile(self.file_path):
#                 print(f"Reading Label data from file {self.file_path}")
#                 with open(self.file_path, 'rb') as f:
#                     label_data = pickle.load( f )
#                     if label_data:
#                         self.names = label_data[0]
#                         self.colors = label_data[1]
#                         self.markers = label_data[2]
#         except Exception as err:
#             print( f" Can't read markers: {err}" )


class SpatialDataManager(ModeDataManager):

    def __init__( self  ):   # Tile shape (y,x) matches image shape (row,col)
        super(SpatialDataManager, self).__init__()
        from spectraclass.data.spatial.tile.manager import TileManager
        self.tiles = TileManager.instance()

    @classmethod
    def extent(cls, image_data: xa.DataArray ) -> List[float]: # left, right, bottom, top
        xc, yc = image_data.coords[image_data.dims[-1]].values, image_data.coords[image_data.dims[-2]].values
        dx2, dy2 = (xc[1]-xc[0])/2, (yc[0]-yc[1])/2
        return [ xc[0]-dx2,  xc[-1]+dx2,  yc[-1]-dy2,  yc[0]+dy2 ]

    def getConstantXArray(self, fill_value: float, shape: Tuple[int], dims: Tuple[str], **kwargs) -> xa.DataArray:
        coords = kwargs.get( "coords", { dim: np.arange(shape[id]) for id, dim in enumerate(dims) } )
        result: xa.DataArray = xa.DataArray( np.full( shape, fill_value ), dims=dims, coords=coords )
        result.attrs.update( kwargs.get("attrs",{}) )
        result.name = kwargs.get( "name", "")
        return result

    def reduce(self, data: xa.DataArray):
        from spectraclass.reduction.embedding import ReductionManager
        if self.reduce_method != "":
            dave, dmag =  data.values.mean(0), 2.0*data.values.std(0)
            normed_data = ( data.values - dave ) / dmag
            reduced_spectra, reproduction = ReductionManager.instance().reduce( normed_data, self.reduce_method, self.model_dims, self.reduce_nepochs, self.reduce_sparsity )
            coords = dict( samples=data.coords['samples'], band=np.arange( self.model_dims )  )
            return xa.DataArray( reduced_spectra, dims=['samples', 'band'], coords=coords )
        return data

    @classmethod
    def getRGB(cls, raster_data: xa.DataArray ) -> xa.DataArray:
        b = raster_data.isel( band=slice( 13, 27 ) ).mean(dim="band", skipna=True)
        g = raster_data.isel( band=slice( 29, 44 ) ).mean(dim="band", skipna=True)
        r = raster_data.isel( band=slice( 51, 63 ) ).mean(dim="band", skipna=True)
        rgb: xa.DataArray = xa.concat( [r,g,b], 'band' )
        return cls.scale_to_bounds( rgb, (0, 1) ).transpose('y', 'x', 'band')

    def markerFileName(self) -> str:
        try:
            return self.tiles.image_name.strip("/").replace("/","_")
        except Exception:
            return ""

    @classmethod
    def scale_to_bounds(cls, raster: xa.DataArray, bounds: Tuple[float, float] ) -> xa.DataArray:
        vmin = raster.min( dim=raster.dims[:2], skipna=True )
        vmax = raster.max(dim=raster.dims[:2], skipna=True )
        scale = (bounds[1]-bounds[0])/(vmax-vmin)
        return  (raster - vmin)*scale + bounds[0]

    @classmethod
    def norm_to_bounds(cls, raster: xa.DataArray, dims: Tuple[str, str], bounds: Tuple[float, float], stretch: float ) -> xa.DataArray:
        scale = ( ( bounds[1] - bounds[0] ) * stretch ) / raster.std(dim=['x', 'y'])
        return  ( raster - raster.mean(dim=dims) ) * scale + (( bounds[1] + bounds[0] )/2.0)

    @classmethod
    def unit_norm(cls, raster: xa.DataArray, dim: List[str] ):
        std: xa.DataArray = raster.std(dim=dim, skipna=True)
        meanval: xa.DataArray = raster.mean(dim=dim, skipna=True)
        unit_centered: xa.DataArray = ( ( raster - meanval ) / std ) + 0.5
        unit_centered = unit_centered.where( unit_centered > 0, 0 )
        unit_centered = unit_centered.where(unit_centered < 1, 1 )
        return unit_centered

    @classmethod
    def normalize(cls, raster: xa.DataArray, scale = 1.0, center = True ):
        std = raster.std(dim=['x','y'], skipna=True)
        if center:
            meanval = raster.mean(dim=['x','y'], skipna=True)
            centered= raster - meanval
        else:
            centered = raster
        result =  centered * scale / std
        result.attrs = raster.attrs
        return result



    @classmethod
    def raster2points(cls, raster: xa.DataArray ) -> xa.DataArray:
        stacked_raster = raster.stack(samples=raster.dims[-2:]).transpose()
        if np.issubdtype( raster.dtype, np.integer ):
            nodata = stacked_raster.attrs.get('_FillValue',-2)
            point_data = stacked_raster.where( stacked_raster != nodata, drop=True ).astype(np.int32)
        else:
            point_data = stacked_raster.dropna(dim='samples', how='any')
        print(f" raster2points -> [{raster.name}]: Using {point_data.shape[0]} valid samples out of {stacked_raster.shape[0]} pixels")
        point_data.attrs['dsid'] = raster.name
        return point_data

    @classmethod
    def get_color_bounds( cls, raster: xa.DataArray ):
        colorstretch = 1.25
        ave = raster.mean(skipna=True).values
        std = raster.std(skipna=True).values
        return dict( vmin= ave - std*colorstretch, vmax= ave + std*colorstretch  )

    @classmethod
    def plotRaster(cls, raster: xa.DataArray, **kwargs ):
        from matplotlib.colorbar import Colorbar
        ax = kwargs.pop( 'ax', None )
        showplot = ( ax is None )
        if showplot: fig, ax = plt.subplots(1,1)
        colors = kwargs.pop('colors', None )
        title = kwargs.pop( 'title', raster.name )
        rescale = kwargs.pop( 'rescale', None )
        colorbar = kwargs.pop( 'colorbar', True )
        x = raster.coords[ raster.dims[1] ].values
        y = raster.coords[ raster.dims[0] ].values
        try:
            xstep = (x[1] - x[0]) / 2.0
        except IndexError: xstep = .1
        try:
            ystep = (y[1] - y[0]) / 2.0
        except IndexError: ystep = .1
        left, right = x[0] - xstep, x[-1] + xstep
        bottom, top = y[-1] + ystep, y[0] - ystep
        defaults = dict( origin= 'upper', interpolation= 'nearest' )
        defaults["alpha"] = kwargs.get( "alpha", 1.0 )
        cbar_kwargs = {}
        if colors is  None:
            defaults.update( dict( cmap="jet" ) )
        else:
            rgbs = [ cval[2] for cval in colors ]
            cmap: ListedColormap = ListedColormap( rgbs )
            color_values = [ float(cval[0]) for cval in colors]
            color_bounds = get_color_bounds(color_values)
            norm = mpl.colors.BoundaryNorm( color_bounds, len( colors )  )
            cbar_kwargs.update( dict( cmap=cmap, norm=norm, boundaries=color_bounds, ticks=color_values, spacing='proportional' ) )
            defaults.update( dict( cmap=cmap, norm=norm ) )
        if not hasattr(ax, 'projection'): defaults['aspect'] = 'auto'
        vrange = kwargs.pop( 'vrange', None )
        if vrange is not None:
            defaults['vmin'] = vrange[0]
            defaults['vmax'] = vrange[1]
        if (colors is  None) and ("vmax" not in defaults):
            defaults.update( cls.get_color_bounds( raster ) )
        defaults.update(kwargs)
        if defaults['origin'] == 'upper':   defaults['extent'] = [left, right, bottom, top]
        else:                               defaults['extent'] = [left, right, top, bottom]
        if rescale is not None:
            raster = cls.scale_to_bounds(raster, rescale)
        img = ax.imshow( raster.data, zorder=1, **defaults )
        ax.set_title(title)
        if colorbar and (raster.ndim == 2):
            cbar: Colorbar = ax.figure.colorbar(img, ax=ax, **cbar_kwargs )
            if colors is not None:
                cbar.set_ticklabels( [ cval[1] for cval in colors ] )
        if showplot: plt.show()
        return img

    def getXarray(self, id: str, xcoords: Dict, subsample: int, xdims: OrderedDict, **kwargs) -> xa.DataArray:
        np_data: np.ndarray = SpatialDataManager.instance().getInputFileData(id, subsample, tuple(xdims.keys()))
        dims, coords = [], {}
        for iS in np_data.shape:
            coord_name = xdims[iS]
            dims.append(coord_name)
            coords[coord_name] = xcoords[coord_name]
        attrs = {**kwargs, 'name': id}
        return xa.DataArray(np_data, dims=dims, coords=coords, name=id, attrs=attrs)

    def prepare_inputs(self, *args, **kwargs ):
        from spectraclass.data.spatial.tile.tile import Block, DataType
        block: Block = self.tiles.getBlock( )
        ( point_data, point_coords ) = block.getPointData( dstype = DataType.Embedding, subsample = self.subsample )
        dsid = point_data.attrs['dsid']
        model_coords = dict( samples=point_data.samples, model=np.arange(self.model_dims) )
        data_vars = dict( raw=point_data )
        if self.reduce_method != "":
            reduced_spectra, reproduction = ReductionManager.instance().reduce( point_data, self.reduce_method, self.model_dims, self.reduce_nepochs, self.reduce_sparsity )
            data_vars['reduction'] = xa.DataArray( reduced_spectra, dims=['samples', 'model'], coords=model_coords )
            data_vars['reproduction'] = point_data.copy( data=reproduction )
        dataset = xa.Dataset( data_vars ) # , attrs={'type': 'spectra'} )
        file_name_base = dsid if self.reduce_method == "None" else f"{dsid}-{self.reduce_method}-{self.model_dims}"
        self.dataset = f"{file_name_base}-ss{self.subsample}" if self.subsample > 1 else file_name_base
        output_file = os.path.join(self.datasetDir, self.dataset + ".nc")
        outputDir = os.path.join( self.cache_dir, dm().project_name )
        os.makedirs(outputDir, 0o777, True)
        print(f"Writing output to {output_file}")
        if os.path.exists(output_file): os.remove(output_file)
        dataset.to_netcdf(output_file)


