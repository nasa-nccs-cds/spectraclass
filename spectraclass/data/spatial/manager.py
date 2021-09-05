import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
from spectraclass.gui.control import UserFeedbackManager, ufm
from spectraclass.reduction.embedding import ReductionManager, rm
from spectraclass.data.base import ModeDataManager
import matplotlib.pyplot as plt
from collections import OrderedDict
from spectraclass.util.logs import LogManager, lgm, exception_handled
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.data.spatial.tile.tile import Block
import os, math, pickle
import rioxarray as rio
from .modes import *


def dm():
    from spectraclass.data.base import DataManager
    return DataManager.instance()


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
    colorstretch = 1.25

    def __init__( self  ):   # Tile shape (y,x) matches image shape (row,col)
        super(SpatialDataManager, self).__init__()
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        self.tiles: TileManager = tm()

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
        if self.reduce_method and (self.reduce_method.lower() != "none"):
            dave, dmag =  data.values.mean(0), 2.0*data.values.std(0)
            normed_data = ( data.values - dave ) / dmag
            reduced_spectra, reproduction, _ = rm().reduce( normed_data, None, self.reduce_method, self.model_dims, self.reduce_nepochs, self.reduce_sparsity )[0]
            coords = dict( samples=data.coords['samples'], band=np.arange( self.model_dims )  )
            return xa.DataArray( reduced_spectra, dims=['samples', 'band'], coords=coords )
        return data

    def setDatasetId(self, dsid: str):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        toks = dsid.split("_b-")
        block_toks = toks[1].split("-")
        tm().block_shape = [ int(block_toks[0]), int(block_toks[1]) ]
        tm().block_index = [ int(block_toks[2]), int(block_toks[3]) ]
        lgm().log( f"Setting block index to {tm().block_index}, shape = {tm().block_shape}")

    def dsid(self, **kwargs) -> str:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block = kwargs.get( 'block', tm().getBlock() )
        reduction_method = f"raw" if self.reduce_method.lower() == "none" else f"{self.reduce_method}-{self.model_dims}"
        file_name_base = "-".join( [block.file_name, reduction_method] )
        return f"{file_name_base}-ss{self.subsample}" if self.subsample > 1 else file_name_base

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
    def raster2points(cls, base_raster: xa.DataArray ) -> xa.DataArray:   #  base_raster dims: [ band, y, x ]
        stacked_raster = base_raster.stack(samples=base_raster.dims[-2:]).transpose()
        if np.issubdtype( base_raster.dtype, np.integer ):
            nodata = stacked_raster.attrs.get('_FillValue',-2)
            point_data = stacked_raster.where( stacked_raster != nodata, drop=True ).astype(np.int32)
        else:
            point_data = stacked_raster.dropna(dim='samples', how='any')
        lgm().log(f" raster2points -> [{base_raster.name}]: Using {point_data.shape[0]} valid samples out of {stacked_raster.shape[0]} pixels")
        point_data.attrs['dsid'] = base_raster.name
        return point_data

    @classmethod
    def addTextureBands(cls, base_raster: xa.DataArray ) -> xa.DataArray:   #  base_raster dims: [ band, y, x ]
        from spectraclass.features.texture.manager import TextureManager, texm
        return texm().addTextureBands( base_raster )

    @classmethod
    def get_color_bounds( cls, raster: xa.DataArray ):
        ave = raster.mean(skipna=True).values
        std = raster.std(skipna=True).values
        if std == 0.0:
            msg =  "This block does not appear to contain any data.  Suggest trying a different tile/block."
            ufm().show( msg, "red" ); lgm().log( "\n" +  msg + "\n"  )
        return dict( vmin= ave - std * cls.colorstretch, vmax= ave + std * cls.colorstretch  )

    @classmethod
    def plotRaster(cls, raster: xa.DataArray, **kwargs ):
        from matplotlib.colorbar import Colorbar
        from spectraclass.application.controller import app
        ax = kwargs.pop( 'ax', None )
        showplot = ( ax is None )
        if showplot: fig, ax = plt.subplots(1,1)
        itype = kwargs.pop('itype', 'base' )
        title = kwargs.pop( 'title', raster.name )
        zeros = kwargs.pop('zeros', False)
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
        if itype ==  'base':
            defaults.update( dict( cmap=app().color_map ) )
        else:
            cspecs = lm().get_labels_colormap()
            cbar_kwargs.update( cspecs )
            defaults.update(  cmap=cspecs['cmap'], norm=cspecs['norm'] )
        if not hasattr(ax, 'projection'): defaults['aspect'] = 'auto'
        vrange = kwargs.pop( 'vrange', None )
        if vrange is not None:
            defaults['vmin'] = vrange[0]
            defaults['vmax'] = vrange[1]
        if (itype ==  'base') and ("vmax" not in defaults):
            defaults.update( cls.get_color_bounds( raster ) )
        defaults.update(kwargs)
        if defaults['origin'] == 'upper':   defaults['extent'] = [left, right, bottom, top]
        else:                               defaults['extent'] = [left, right, top, bottom]
        if rescale is not None:
            raster = cls.scale_to_bounds(raster, rescale)
        lgm().log( f"$$$COLOR: Plotting tile image with parameters: {defaults}")
        img_data = raster.data if not zeros else np.zeros( raster.shape, np.int )
        img = ax.imshow( img_data, zorder=1, **defaults )
        ax.set_title(title)
        if colorbar:
            cbar: Colorbar = ax.figure.colorbar(img, ax=ax, **cbar_kwargs )
            cbar.set_ticklabels( [ cval[1] for cval in lm().labeledColors ] )
 #       if showplot: plt.show()
        return img

    @classmethod
    def plotOverlayImage(cls, raster: np.ndarray, ax, extent: Tuple[float], **kwargs):
        defaults = dict( origin= 'upper',  alpha = 0.5, extent=extent )
        if not hasattr(ax, 'projection'): defaults['aspect'] = 'auto'
        defaults.update(kwargs)
        lgm().log( f"\n $$$COLOR: Plotting overlay image with parameters: {defaults}, data range = {[raster.min(),raster.max()]}, raster type = {raster.dtype}, shape = {raster.shape}\n")
        img = ax.imshow( raster, zorder=2, **defaults )
        return img

    def getXarray(self, id: str, xcoords: Dict, subsample: int, xdims: OrderedDict, **kwargs) -> xa.DataArray:
        np_data: np.ndarray = SpatialDataManager.instance().getInputFileData(id)
        dims, coords = [], {}
        for iS in np_data.shape:
            coord_name = xdims[iS]
            dims.append(coord_name)
            coords[coord_name] = xcoords[coord_name]
        attrs = {**kwargs, 'name': id}
        return xa.DataArray(np_data, dims=dims, coords=coords, name=id, attrs=attrs)

    def reduced_dataset_name(self, dsid: str ):
        file_name_base = f"{dsid}-raw" if self.reduce_method.lower() == "none" else f"{dsid}-{self.reduce_method}-{self.model_dims}"
        return f"{file_name_base}-ss{self.subsample}" if self.subsample > 1 else file_name_base

    @exception_handled
    def prepare_inputs(self, *args, **kwargs ):
        tile_data = self.tiles.getTileData()
        for block in self.tiles.tile.getBlocks():
            block.clearBlockCache()
#           block.addTextureBands( )
            blocks_point_data: xa.DataArray = block.getPointData()[0]
            if blocks_point_data.size == 0:
               lgm().log( f" Warning:  Block {block.block_coords} has no valid samples.", print=True )
            else:
                prange = [ blocks_point_data.values.min(), blocks_point_data.values.max() ]
                lgm().log(f" Preparing point data with shape {blocks_point_data.shape} and range = {prange}", print=True)
                blocks_reduction = rm().reduce( blocks_point_data, None, self.reduce_method, self.model_dims, self.reduce_nepochs, self.reduce_sparsity )
                if blocks_reduction is not None:
                    self.model_dims = blocks_reduction[0][0].shape[1]
                    for ( reduced_spectra, reproduction, point_data ) in blocks_reduction:
                        file_name = point_data.attrs['file_name']
                        model_coords = dict( samples=point_data.samples, model=np.arange(self.model_dims) )
                        raw_data: xa.DataArray = block.data
                        try: raw_data.attrs['wkt'] = tile_data.spatial_ref.crs_wkt
                        except: pass
                        data_vars = dict( raw=raw_data, norm=point_data )
                        reduced_dataArray =  xa.DataArray( reduced_spectra, dims=['samples', 'model'], coords=model_coords )
                        data_vars['reduction'] = reduced_dataArray
                        data_vars['reproduction'] = reproduction
                        result_dataset = xa.Dataset( data_vars )
                        self.dataset = self.reduced_dataset_name( file_name )
                        output_file = os.path.join( self.datasetDir, self.dataset + ".nc" )
                        self._reduced_raster_file = os.path.join(self.datasetDir, self.dataset + ".tif")
                        lgm().log(f" Writing reduced[{self.reduce_scope}] output to {output_file} with {blocks_point_data.size} samples, dset attrs:")
                        for varname, da in result_dataset.data_vars.items():
                            da.attrs['long_name'] = f"{file_name}.{varname}"
                        print( f"Writing output file: '{output_file}' with {blocks_point_data.size} samples")
                        os.makedirs( os.path.dirname( output_file ), exist_ok=True )
                        result_dataset.to_netcdf( output_file )
                        print(f"Writing raster file: '{self._reduced_raster_file}' with dims={reduced_dataArray.dims}, attrs = {reduced_dataArray.attrs}")
#                        reduced_dataArray.rio.set_spatial_dims()
                        raw_data.rio.to_raster( self._reduced_raster_file )


    @property
    def reduced_raster_file(self):
        return self._reduced_raster_file

    def getFilePath(self, use_tile: bool ) -> str:
        base_dir = dm().modal.data_dir
        base_file = self.tiles.tileName() if use_tile else self.tiles.image_name
        if base_file.endswith(".mat") or base_file.endswith(".tif"):
            return f"{base_dir}/{base_file}"
        else:
            return f"{base_dir}/{base_file}.tif"

    def writeGeotiff(self, raster_data: xa.DataArray ) -> Optional[str]:
        output_file = self.getFilePath(True)
        try:
            if os.path.exists(output_file): os.remove(output_file)
            lgm().log(f"Writing (raster) tile file {output_file}")
            raster_data.rio.to_raster(output_file)
            return output_file
        except Exception as err:
            lgm().log(f"Unable to write raster file to {output_file}: {err}")
            return None

    def readSpectralData(self, read_tile: bool) -> xa.DataArray:
        input_file_path = self.getFilePath( read_tile )
        assert os.path.isfile( input_file_path ), f"Input file does not exist: {input_file_path}"
        return self.readDataFile( input_file_path )

    def readDataFile(self, file_path: str ):
        if file_path.endswith(".mat"):
            return self.readMatlabFile( file_path )
        else:
            return self.readGeoTiff( file_path )

    def readGeoTiff(self, input_file_path: str ) -> xa.DataArray:
        input_bands = rio.open_rasterio(input_file_path)
        if 'transform' not in input_bands.attrs.keys():
            gts = input_bands.spatial_ref.GeoTransform.split()
            input_bands.attrs['transform'] = [float(gts[i]) for i in [1, 2, 0, 4, 5, 3]]
            input_bands.attrs['fileformat'] = "tif"
        lgm().log(f"Reading raster file {input_file_path}, dims = {input_bands.dims}, shape = {input_bands.shape}")
        return input_bands

    def readMatlabFile(self, input_file_path: str ) -> xa.DataArray:
        from scipy.io import loadmat
        from spectraclass.gui.spatial.image import toXA
        gtdset = loadmat(input_file_path)
        vnames = [ vid for vid in gtdset.keys() if not vid.startswith("_") ]
        assert len( vnames ) == 1, f"Can't find unique variable in matlab file {input_file_path}, vars = {vnames} "
        gtarray: np.ndarray = gtdset[vnames[0]]
        print(f"Reading variable '{vnames[0]}' from Matlab dataset '{input_file_path}': {gtdset['__header__']}")
        filename, file_extension = os.path.splitext( input_file_path )
        return toXA( vnames[0], gtarray, file_extension, True )

    def getClassMap(self) -> Optional[xa.DataArray]:
        class_file_path = os.path.join( self.data_dir, self.class_file )
        if not os.path.isfile(class_file_path): return None
        print( f"\nReading class file: {class_file_path}\n")
        return self.readDataFile( class_file_path )


