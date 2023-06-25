import numpy as np
import xarray as xa
from spectraclass.reduction.vae.trainer import mt
import numpy.ma as ma
from pathlib import Path
from spectraclass.gui.control import ufm
from spectraclass.reduction.embedding import rm
from spectraclass.data.base import ModeDataManager
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Set
import stat, os, time, json
from spectraclass.data.spatial.tile.tile import Block, Tile
from rioxarray.exceptions import NoDataInBounds
from collections import OrderedDict
from spectraclass.util.logs import lgm, exception_handled, log_timing
from spectraclass.model.labels import lm
import rioxarray as rio

def dm():
    from spectraclass.data.base import DataManager
    return DataManager.instance()

def s2np( value: str ) -> np.ndarray:
    toks: List[str] = value.strip("[]\n").split(",")
    return np.array( [float(tv) for tv in toks] )

class SpatialDataManager(ModeDataManager):
    colorstretch = 1.25

    def __init__( self  ):   # Tile shape (y,x) matches image shape (row,col)
        super(SpatialDataManager, self).__init__()
        self.spectral_means: List[ Tuple[int,np.ndarray] ] = []

    def getSpectralData( self, **kwargs ) -> Optional[xa.DataArray]:
        from spectraclass.gui.spatial.map import MapManager, mm
        return mm().getPointData( **kwargs )

    @exception_handled
    def getModelData( self, raw_model_data: xa.DataArray, **kwargs) -> Optional[xa.DataArray]:
        from spectraclass.gui.spatial.map import MapManager, mm
        tmask: np.ndarray = mm().threshold_mask(raster=False)
        if tmask is None:
            lgm().log(f"#GID: MAP: model_data[{raw_model_data.dims}], shape= {raw_model_data.shape}, attrs={raw_model_data.attrs.keys()},  NO threshold mask")
            result =  raw_model_data
        else:
            result = raw_model_data[tmask]
            result.attrs.update( raw_model_data.attrs )
            lgm().log( f"#GID: MAP: model_data[{raw_model_data.dims}], shape= {raw_model_data.shape}, attrs={raw_model_data.attrs.keys()}, mask shape = {tmask.shape}")
            lgm().log( f"#GID: filtered model_data[{result.dims}], shape= {result.shape}")
        nnan = np.count_nonzero( np.isnan( result.values.sum(axis=1) ) )
        lgm().log(f"#GID:getModelData->  nnan-bands = {nnan}/{result.shape[0]}, attrs = {result.attrs.keys()}")
        return result

    @classmethod
    def extent(cls, image_data: xa.DataArray ) -> List[float]: # left, right, bottom, top
        xc, yc = image_data.coords[image_data.dims[-1]].values, image_data.coords[image_data.dims[-2]].values
        dx2, dy2 = (xc[1]-xc[0])/2, (yc[0]-yc[1])/2
        return [ xc[0]-dx2,  xc[-1]+dx2,  yc[-1]-dy2,  yc[0]+dy2 ]

    def gui(self, **kwargs):
        return None # self.tile_selection_basemap.gui()

    def update_extent(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        (x0, x1, y0, y1) = tm().tile.extent
 #       self.tile_selection_basemap.set_bounds( (x0, x1), (y0, y1) )
 #       self.tile_selection_basemap.update()

    def getConstantXArray(self, fill_value: float, shape: Tuple[int], dims: Tuple[str], **kwargs) -> xa.DataArray:
        coords = kwargs.get( "coords", { dim: np.arange(shape[id]) for id, dim in enumerate(dims) } )
        result: xa.DataArray = xa.DataArray( np.full( shape, fill_value ), dims=dims, coords=coords )
        result.attrs.update( kwargs.get("attrs",{}) )
        result.name = kwargs.get( "name", "")
        return result

    def pnorm( self, data: xa.DataArray, dim: int = 1 ) -> xa.DataArray:
        dave, dmag = np.nanmean( data.values, keepdims=True, axis=dim ), np.nanstd( data.values, keepdims=True, axis=dim )
        normed_data = (data.values - dave) / dmag
        return data.copy( data=normed_data )

    def sum( self, data: xa.DataArray, dim: int = 0 ) -> xa.DataArray:
        result = data.sum( axis=dim )
        result.attrs['scale'] = data.shape[dim]
        return result

    def range( self, data: xa.DataArray, dim: int = 0 ) -> Tuple[xa.DataArray,xa.DataArray]:
        return ( data.min( axis=dim ), data.max( axis=dim ) )

    def dsid(self, **kwargs) -> str:
        from spectraclass.data.spatial.tile.manager import tm
        block = kwargs.get( 'block', tm().getBlock() )
        return f"{block.file_name}-ss{self.subsample_index}" if self.subsample_index > 1 else block.file_name

    @classmethod
    def getRGB(cls, raster_data: xa.DataArray ) -> xa.DataArray:
        b = raster_data.isel( band=slice( 13, 27 ) ).mean(dim="band", skipna=True)
        g = raster_data.isel( band=slice( 29, 44 ) ).mean(dim="band", skipna=True)
        r = raster_data.isel( band=slice( 51, 63 ) ).mean(dim="band", skipna=True)
        rgb: xa.DataArray = xa.concat( [r,g,b], 'band' )
        return cls.scale_to_bounds( rgb, (0, 1) ).transpose('y', 'x', 'band')

    def markerFileName(self) -> str:
        try:
            return dm().modal.image_name.strip("/").replace("/","_")
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
    def addTextureBands(cls, base_raster: xa.DataArray ) -> xa.DataArray:   #  base_raster dims: [ band, y, x ]
        from spectraclass.features.texture.manager import texm
        return texm().addTextureBands( base_raster )

#     @classmethod
#     def plotRaster(cls, raster: xa.DataArray, **kwargs ):
#         from matplotlib.colorbar import Colorbar
#         from spectraclass.application.controller import app
#         ax = kwargs.pop( 'ax', None )
#         showplot = ( ax is None )
#         if showplot: fig, ax = plt.subplots(1,1)
#         itype = kwargs.pop('itype', 'base' )
#         title = kwargs.pop( 'title', raster.name )
#         zeros = kwargs.pop('zeros', False)
# #        rescale = kwargs.pop( 'rescale', None )
#         colorbar = kwargs.pop( 'colorbar', True )
#         defaults = dict( origin= 'upper', interpolation= 'nearest' )
#         defaults["alpha"] = kwargs.get( "alpha", 1.0 )
#         cbar_kwargs = {}
#         if itype ==  'base':
#             defaults.update( dict( cmap=app().color_map ) )
#         else:
#             cspecs = lm().get_labels_colormap()
#             cbar_kwargs.update( cspecs )
#             defaults.update(  cmap=cspecs['cmap'], norm=cspecs['norm'] )
#         if not hasattr(ax, 'projection'): defaults['aspect'] = 'auto'
#         vrange = kwargs.pop( 'vrange', None )
#         if vrange is not None:
#             defaults['vmin'] = vrange[0]
#             defaults['vmax'] = vrange[1]
#         if (itype ==  'base') and ("vmax" not in defaults):
#             defaults.update( cls.get_color_bounds( raster ) )
#         xlim = kwargs.pop('xlim', [] )
#         ylim = kwargs.pop('ylim', [] )
#         defaults['extent'] = xlim + ylim
#         defaults.update(kwargs)
# #        if defaults['origin'] == 'upper':   defaults['extent'] = [left, right, bottom, top]
# #        else:                               defaults['extent'] = [left, right, top, bottom]
# #        if rescale is not None:
# #            raster = cls.scale_to_bounds(raster, rescale)
#         lgm().log( f"$$$COLOR: Plotting tile image with parameters: {defaults}")
#         img_data = raster.data if not zeros else np.zeros( raster.shape, np.int32 )
#         img = ax.imshow( img_data, zorder=1, **defaults )
#         ax.set_title(title)
#         if colorbar:
#             cbar: Colorbar = ax.figure.colorbar(img, ax=ax, **cbar_kwargs )
#             cbar.set_ticklabels( [ cval[1] for cval in lm().labeledColors ] )
#  #       if showplot: plt.show()
#         return img

    @classmethod
    def plotOverlayImage(cls, raster: np.ndarray, ax, extent: Tuple[float], **kwargs):
        defaults = dict( origin= 'upper',  alpha = 0.5, extent=extent )
        if not hasattr(ax, 'projection'): defaults['aspect'] = 'auto'
        defaults.update(kwargs)
        lgm().log( f"\n $$$COLOR: Plotting overlay image with parameters: {defaults}, data range = {[raster.min(),raster.max()]}, raster type = {raster.dtype}, shape = {raster.shape}\n")
        img = ax.imshow( raster, zorder=2.5, **defaults )
        return img

    def reduced_dataset_name(self, dsid: str ):
        file_name_base =  f"{dsid}-{self.model_dims}"
        return f"{file_name_base}-ss{self.subsample_index}" if self.subsample_index > 1 else file_name_base

    def empty_array(self, dims ):
        coords = { d: np.empty([1]) for d in dims }
        data = np.empty([1]*len(dims) )
        return xa.DataArray( data, dims, coords )

    def get_empty_dataset(self) -> xa.Dataset:
        data_vars = dict( raw = self.empty_array(['band', 'y', 'x']),
                          norm = self.empty_array(['samples', 'band']),
                          reduction = self.empty_array(['samples', 'model']),
                          reproduction = self.empty_array(['samples', 'band']) )
        coords = { c: np.empty([1]) for c in [ 'x', 'y', 'band', 'samples', 'model' ] }
        return xa.Dataset( data_vars=data_vars, coords=coords )

    @exception_handled
    def process_block( self, block: Block, has_metadata: bool  ) -> Optional[xa.Dataset]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.base import DataManager, dm
        t0, reprocess = time.time(), tm().reprocess
        block_data_file = dm().modal.dataFile(block=block)
        if os.path.exists(block_data_file) and reprocess:
            os.remove( block_data_file )

        if os.path.exists(block_data_file):
            if not has_metadata:
                lgm().log(f"** Reading BLOCK{block.cindex}: {block_data_file} " )
                block_dataset = xa.open_dataset( block_data_file )
                return block_dataset
            else:
                lgm().log(f"** Skipping processed BLOCK{block.cindex}: {block_data_file} " )
                return None
        else:
            ea1, ea2 = np.empty(shape=[0], dtype=np.float32), np.empty(shape=[0, 0], dtype=np.float32)
            coord_data = {}
            ufm().show( f" *** Processing Block{block.block_coords}" )
            raw_data: Optional[xa.DataArray] = block.data
            result_dataset: Optional[xa.Dataset] = None
            if raw_data is not None:
                try:
                    blocks_point_data, coord_data = block.getPointData(norm=False,anomaly="none")
                    lgm().log(f"** BLOCK{block.cindex}: Read point data, shape = {blocks_point_data.shape}, dims = {blocks_point_data.dims}")
                except NoDataInBounds:
                    blocks_point_data = xa.DataArray(ea2, dims=('samples', 'band'), coords=dict(samples=ea1, band=ea1))

                if blocks_point_data.size == 0:
                    ufm().show(f" *** NO DATA in BLOCK {block.block_coords} *** ")
                    return None
                smean = np.nanmean( block.raw_point_data.values, axis=0 )
                ptcount = np.count_nonzero( ~np.isnan(block.raw_point_data) )
                self.spectral_means.append( ( ptcount, smean ) )
                data_vars = dict( raw=raw_data )
                lgm().log(  f" Writing output file: '{block_data_file}' with {blocks_point_data.shape[0]} samples" )
                data_vars['mask'] = xa.DataArray( coord_data['mask'].reshape(raw_data.shape[1:]), dims=['y', 'x'], coords={d: raw_data.coords[d] for d in ['x', 'y']} )
                result_dataset = xa.Dataset(data_vars)
                result_dataset.attrs['tile_shape'] = tm().tile.data.shape
                result_dataset.attrs['block_dims'] = tm().block_dims
                result_dataset.attrs['tile_size'] = tm().tile_size
                result_dataset.attrs['nsamples'] = blocks_point_data.shape[0]
                result_dataset.attrs['nbands'] = blocks_point_data.shape[1]
                result_dataset.attrs['valid_bands'] = str(dm().valid_bands())
                for (aid, aiv) in tm().tile.data.attrs.items():
                    if aid not in result_dataset.attrs:
                        result_dataset.attrs[aid] = aiv
                lgm().log( f" Writing preprocessed output to {block_data_file} with {blocks_point_data.size} samples, dset attrs:")
                for varname, da in result_dataset.data_vars.items():
                    da.attrs['long_name'] = ".".join([block.file_name, varname])
                for vname, v in data_vars.items():
                    lgm().log( f" ---> {vname}: shape={v.shape}, size={v.size}, dims={v.dims}, coords={[':'.join([cid, str(c.shape)]) for (cid, c) in v.coords.items()]}")
                write_dir = os.path.dirname(block_data_file)
                open_perm = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO
                os.makedirs(write_dir, exist_ok=True)
                os.chmod( write_dir, open_perm )
                result_dataset.to_netcdf(block_data_file)
                os.chmod( block_data_file, open_perm )
                lgm().log( f" ---------  FINISHED PROCESSING BLOCK {block.block_coords} in {time.time()-t0:.2f} sec ---------  ")
            return result_dataset

    def get_scaling( self, sums: List[xa.DataArray] ) -> xa.DataArray:
        dsum: xa.DataArray = None
        npts = 0
        for sum in sums:
            if dsum is None:
                dsum = sum
                npts = dsum.attrs['scale']
            else:
                dsum = dsum + sum
                npts = npts + sum.attrs['scale']
        return dsum/npts

    @exception_handled
    def prepare_inputs(self, **kwargs ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        tm().autoprocess = False
        attrs, block_sizes = {}, {}
        nbands = None
        lgm().log(f" Preparing inputs, reprocess={tm().reprocess}", print=True)
        blocks = tm().tile.getBlocks()
        try:
            has_metadata = (self.metadata is not None)
            for image_index in range( dm().modal.num_images ):
                self.set_current_image( image_index )
                action = "Preprocessing data blocks" if tm().reprocess else "Processing metadata"
                lgm().log(f" {action} for image {dm().modal.image_name} with {len(blocks)} blocks.", print=True)
                ufm().show( f"{action} for image {dm().modal.image_name}" )
                for block in blocks:
                    result_dataset: xa.Dataset = self.process_block( block, has_metadata )
                    if result_dataset is not None:
                        block_sizes[ block.cindex ] = result_dataset.attrs[ 'nsamples']
                        if nbands is None: nbands = result_dataset.attrs[ 'nbands']
                        result_dataset.close()
            if len( self.spectral_means ) > 0:
                self.process_spectral_mean()
            if not has_metadata:
                self.write_metadata(block_sizes, attrs)
            mt().train()
        except Exception as err:
            print( f"\n *** Error in processing workflow, check log file for details: {lgm().log_file} *** ")
            lgm().exception("prepare_inputs error:")

    @exception_handled
    def process_spectral_mean(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        total_samples = sum( [ v[0] for v in self.spectral_means ]  )
        wmeans = [ (v[0]/total_samples)*v[1] for v in self.spectral_means ]
        spectral_mean = xa.DataArray( np.add.reduce( wmeans ), dims=['band'], coords=dict( band=tm().tile.data.band ) )
        file_path = f"{dm().cache_dir}/{self.modelkey}.spectral_mean.nc"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            if os.path.isfile(file_path): os.remove(file_path)
            spectral_mean.to_netcdf( file_path )
            lgm().log(f"Writing spectral_mean file: {file_path}", print=True)
        except Exception as err:
            lgm().log(f" ---> ERROR Writing spectral_mean file at {file_path}: {err}", print=True)


    def update_band_range(self, band_range: Tuple[np.ndarray,np.ndarray], dataset: xa.Dataset ) -> Tuple[np.ndarray,np.ndarray]:
        band_min: np.ndarray = dataset['band_min'].squeeze() if band_range is None else np.fmin( band_range[0], dataset['band_min'].values ).squeeze()
        band_max: np.ndarray = dataset['band_max'].squeeze() if band_range is None else np.fmax( band_range[1], dataset['band_max'].values ).squeeze()
        return ( np.expand_dims(band_min,0), np.expand_dims(band_max,0) )


    def dataFile( self, **kwargs ):
        from spectraclass.data.spatial.tile.tile import Block
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        filter_sig = tm().get_band_filter_signature()
        block: Block = kwargs.get('block',None)
        bindex = tm().block_index if (block is None) else block.block_coords
        file_name = f"{tm().tileName(**kwargs)}-{tm().block_size}-{filter_sig}-{bindex[0]}-{bindex[1]}"
        return os.path.join( self.datasetDir, file_name + f"{self.ext}.nc" )

    def getFilePath(self) -> str:
        filepath = self.image_path( self.image_name )
        return filepath

    def writeGeotiff(self, raster_data: xa.DataArray ) -> Optional[str]:
        output_file = self.getFilePath()
        try:
            if os.path.exists(output_file): os.remove(output_file)
            lgm().log(f"Writing (raster) tile file {output_file}")
            raster_data.rio.to_raster(output_file)
            return output_file
        except Exception as err:
            lgm().log(f"Unable to write raster file to {output_file}: {err}")
            return None

    def readSpectralData(self, **kwargs) -> xa.DataArray:
        input_file_path = self.getFilePath()
        assert os.path.isfile( input_file_path ), f"Input file does not exist: {input_file_path}"
        spectral_data = self.readDataFile( input_file_path, **kwargs )
        return spectral_data

    def readDataFile(self, file_path: str, **kwargs  ):
        # if file_path.endswith(".mat"):
        #     return self.readMatlabFile( file_path )
        # else:
        return self.readGeoTiff( file_path, **kwargs )

    @exception_handled
    def readGeoTiff(self, input_file_path: str ) -> xa.DataArray:
        input_bands = rio.open_rasterio( input_file_path )
        input_bands.attrs['long_name'] = Path(input_file_path).stem
        lgm().log( f"Completed Reading raster file {input_file_path}, dims = {input_bands.dims}, shape = {input_bands.shape}", print=True )
        gt = [ float(sval) for sval in input_bands.spatial_ref.GeoTransform.split() ]
        input_bands.attrs['transform'] = [ gt[1], gt[2], gt[0], gt[4], gt[5], gt[3] ]
        lgm().log(f" --> transform: {input_bands.attrs['transform']}")
        lgm().log(f" --> coord shapes:")
        for (k, v) in input_bands.coords.items(): lgm().log(f"     ** {k}: {v.shape}, range: {(v.values.min(),v.values.max())}")
#        lgm().log("ATTRIBUTES:")
#        for (k,v) in input_bands.attrs.items(): lgm().log(f" ** {k}: {v}" )
#         nodata = input_bands.attrs.get( '_FillValue' )
#         if (nodata is not None) and not np.isnan( nodata ):
#             input_bands = input_bands.where( input_bands != nodata, np.nan )
#             input_bands.attrs['_FillValue'] = np.nan
        return input_bands

    # def readMatlabFile(self, input_file_path: str ) -> xa.DataArray:
    #     from scipy.io import loadmat
    #     from spectraclass.gui.spatial.image import toXA
    #     gtdset = loadmat(input_file_path)
    #     vnames = [ vid for vid in gtdset.keys() if not vid.startswith("_") ]
    #     assert len( vnames ) == 1, f"Can't find unique variable in matlab file {input_file_path}, vars = {vnames} "
    #     gtarray: np.ndarray = gtdset[vnames[0]]
    #     print(f"Reading variable '{vnames[0]}' from Matlab dataset '{input_file_path}': {gtdset['__header__']}")
    #     filename, file_extension = os.path.splitext( input_file_path )
    #     return toXA( vnames[0], gtarray, file_extension, True )

    def getClassMap(self) -> Optional[xa.DataArray]:
        class_file_path = os.path.join( ModeDataManager.data_dir, self.class_file )
        if not os.path.isfile(class_file_path): return None
        print( f"\nReading class file: {class_file_path}\n")
        return self.readDataFile( class_file_path )


