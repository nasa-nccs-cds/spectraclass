import traceback

import numpy as np
import xarray as xa
import ipywidgets as ip
import numpy.ma as ma
from pathlib import Path
from spectraclass.gui.control import ufm
from spectraclass.reduction.embedding import rm
from spectraclass.data.base import ModeDataManager
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Set
import matplotlib.pyplot as plt
import os, time
from rioxarray.exceptions import NoDataInBounds
from collections import OrderedDict
from spectraclass.util.logs import lgm, exception_handled, log_timing
from spectraclass.model.labels import lm
import rioxarray as rio

def dm():
    from spectraclass.data.base import DataManager
    return DataManager.instance()

class SpatialDataManager(ModeDataManager):
    colorstretch = 1.25

    def __init__( self  ):   # Tile shape (y,x) matches image shape (row,col)
        from spectraclass.gui.spatial.basemap import TileServiceBasemap
        super(SpatialDataManager, self).__init__()
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        self.tiles: TileManager = tm()
        self._tile_selection_basemap: TileServiceBasemap = None

    @classmethod
    def extent(cls, image_data: xa.DataArray ) -> List[float]: # left, right, bottom, top
        xc, yc = image_data.coords[image_data.dims[-1]].values, image_data.coords[image_data.dims[-2]].values
        dx2, dy2 = (xc[1]-xc[0])/2, (yc[0]-yc[1])/2
        return [ xc[0]-dx2,  xc[-1]+dx2,  yc[-1]-dy2,  yc[0]+dy2 ]

    def gui(self, **kwargs):
        from spectraclass.gui.spatial.basemap import TileServiceBasemap
        if self._tile_selection_basemap is None:
            self._tile_selection_basemap = TileServiceBasemap( block_selection=True )
            (x0, x1, y0, y1) = self.tiles.tile.extent
            self._tile_selection_basemap.setup_plot( "Tile Selection", (x0, x1), (y0, y1), index=99, size=(3,3), slider=False, **kwargs )
        return self._tile_selection_basemap.gui()

    def update_extent(self):
        (x0, x1, y0, y1) = self.tiles.tile.extent
        self._tile_selection_basemap.set_bounds( [x0, x1], [y0, y1] )
        self._tile_selection_basemap.update()

    def getConstantXArray(self, fill_value: float, shape: Tuple[int], dims: Tuple[str], **kwargs) -> xa.DataArray:
        coords = kwargs.get( "coords", { dim: np.arange(shape[id]) for id, dim in enumerate(dims) } )
        result: xa.DataArray = xa.DataArray( np.full( shape, fill_value ), dims=dims, coords=coords )
        result.attrs.update( kwargs.get("attrs",{}) )
        result.name = kwargs.get( "name", "")
        return result

    def pnorm( self, data: xa.DataArray, dim: int = 1 ) -> xa.DataArray:
        dave, dmag = data.values.mean( dim, keepdims=True ), data.values.std( dim, keepdims=True )
        normed_data = (data.values - dave) / dmag
        return data.copy( data=normed_data )

    def reduce(self, data: xa.DataArray):
        if self.reduce_method and (self.reduce_method.lower() != "none"):
            normed_data = self.pnorm( data )
            reduced_spectra, reproduction, _ = rm().reduce( normed_data, None, self.reduce_method, self.model_dims, self.reduce_nepochs, self.reduce_sparsity )[0]
            coords = dict( samples=data.coords['samples'], band=np.arange( self.model_dims )  )
            return xa.DataArray( reduced_spectra, dims=['samples', 'band'], coords=coords )
        return data

    def setDatasetId(self, dsid: str):
        from spectraclass.data.spatial.tile.manager import tm
        toks = dsid.split("_b-")
        block_toks = toks[1].split("-")
        tm().block_shape = [ int(block_toks[0]), int(block_toks[1]) ]
        tm().block_index = [ int(block_toks[2]), int(block_toks[3]) ]
        lgm().log( f"Setting block index to {tm().block_index}, shape = {tm().block_shape}")

    def dsid(self, **kwargs) -> str:
        from spectraclass.data.spatial.tile.manager import tm
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

    @classmethod
    def get_color_bounds( cls, raster: xa.DataArray ):
        ave = raster.mean(skipna=True).values
        std = raster.std(skipna=True).values
        if std == 0.0:
            msg =  "This block does not appear to contain any data.  Suggest trying a different tile/block."
            ufm().show( msg, "red" ); lgm().log( "\n" +  msg + "\n"  )
        return dict( vmin= ave - std * cls.colorstretch, vmax= ave + std * cls.colorstretch  )

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
#         img_data = raster.data if not zeros else np.zeros( raster.shape, np.int )
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
    def prepare_inputs(self, **kwargs ) -> Dict[Tuple,int]:
        lgm().log(f" Preparing inputs", print=True)
        reprocess = kwargs.get( 'reprocess',False )
        block_nsamples = {}
        ufm().show( f"Preprocessing data blocks for image {dm().modal.image_name}", "blue" )
        ea1, ea2 = np.empty(shape=[0], dtype=np.float), np.empty(shape=[0, 0], dtype=np.float)
        for block in self.tiles.tile.getBlocks():
            block_data_file =  dm().modal.dataFile(block=block)
            nsamples = 0
            coord_data = {}
            process_dataset = True
            block_file_exists = os.path.isfile( block_data_file )
            if block_file_exists:
                dataset: xa.Dataset = xa.open_dataset( block_data_file )
                try:
                    nsamples = 0 if (len( dataset.coords ) == 0) else dataset.coords['samples'].size
                    block_nsamples[block.block_coords] = nsamples
                    process_dataset = reprocess
                except Exception as err:
                    lgm().log(f" Error getting samples from existing block_data_file: {block_data_file}\n ---> ERROR = {err}",  print=True)
            if not process_dataset:
                lgm().log( f" Skipping existing block{block.block_coords} with nsamples={nsamples}, existing file: {block_data_file}", print=True)
            else:
                lgm().log(f" Processing Block{block.block_coords}, shape = {block.shape}",  print=True)
                try:
                    blocks_point_data, coord_data = block.getPointData()
                    lgm().log(f" Read point data, shape = {blocks_point_data.shape}, dims = {blocks_point_data.dims}", print=True)
                except NoDataInBounds:
                    blocks_point_data = xa.DataArray( ea2, dims=('samples','band'), coords = dict(samples=ea1,band=ea1) )

                if blocks_point_data.size > 0:
                    normed_data: xa.DataArray = self.pnorm(blocks_point_data)
                    prange = ( normed_data.values.min(), normed_data.values.max(), normed_data.values.mean() )
                    lgm().log(f" Preparing point data with shape {normed_data.shape}, range={prange}", print=True)
                    blocks_reduction = rm().reduce( normed_data, None, self.reduce_method, self.model_dims, self.reduce_nepochs, self.reduce_sparsity )
                else:
                    em2 = np.empty(shape=[0,self.model_dims], dtype=np.float)
                    reduced_spectra = xa.DataArray(em2, dims=( 'samples', 'model' ), coords=dict(samples=ea1, model=np.arange(self.model_dims)))
                    blocks_reduction = [ ( reduced_spectra, blocks_point_data, blocks_point_data ), ]

                if blocks_reduction is not None:
                    self.model_dims = blocks_reduction[0][0].shape[1]
                    for ( reduced_spectra, reproduction, point_data ) in blocks_reduction:
                        model_coords = dict( samples=point_data.samples, model=np.arange(self.model_dims) )
                        raw_data: xa.DataArray = block.data
                        data_vars = dict( raw=raw_data, norm=point_data )
                        block_nsamples[block.block_coords] = point_data.shape[0]
                        reduced_dataArray =  xa.DataArray( reduced_spectra, dims=['samples', 'model'], coords=model_coords )
                        data_vars['reduction'] = reduced_dataArray
                        data_vars['reproduction'] = reproduction
                        data_vars['mask'] = coord_data['mask']
                        result_dataset = xa.Dataset( data_vars )
#                       self._reduced_raster_file = os.path.join(self.datasetDir, self.dataset + ".tif")
                        lgm().log(f" Writing reduced output to {block_data_file} with {blocks_point_data.size} samples, dset attrs:")
                        for varname, da in result_dataset.data_vars.items():
                            da.attrs['long_name'] = ".".join( [ point_data.attrs['file_name'], varname ] )
                        print( f"Writing output file: '{block_data_file}' with {blocks_point_data.size} samples")
                        if os.path.exists( block_data_file ): os.remove( block_data_file )
                        else: os.makedirs( os.path.dirname( block_data_file ), exist_ok=True )
                        result_dataset.to_netcdf( block_data_file )
#                        print(f"Writing raster file: '{self._reduced_raster_file}' with dims={reduced_dataArray.dims}, attrs = {reduced_dataArray.attrs}")
#                        reduced_dataArray.rio.set_spatial_dims()
#                        raw_data.rio.to_raster( self._reduced_raster_file )
        return block_nsamples

    def getFilePath(self) -> str:
        base_dir = dm().modal.data_dir
        base_file = dm().modal.image_name
        if base_file.endswith(".mat") or base_file.endswith(".tif"):
            return f"{base_dir}/{base_file}"
        else:
            return f"{base_dir}/{base_file}.tif"

    def getMetadataFilePath(self) -> str:
        base_dir = dm().modal.data_dir
        base_file = dm().modal.image_name
        if base_file.endswith(".mat") or base_file.endswith(".tif"):
            return f"{base_dir}/{base_file[:-4]}.mdata.txt"
        else:
            return f"{base_dir}/{base_file}.mdata.txt"

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
        if file_path.endswith(".mat"):
            return self.readMatlabFile( file_path )
        else:
            return self.readGeoTiff( file_path, **kwargs )

    @exception_handled
    def readGeoTiff(self, input_file_path: str, **kwargs ) -> xa.DataArray:
        t0 = time.time()
        raster = rio.open_rasterio( input_file_path, driver='GTiff' )
        input_bands = raster
        input_bands.attrs['long_name'] = Path(input_file_path).stem
        lgm().log( f"Completed Reading raster file {input_file_path}, dims = {input_bands.dims}, shape = {input_bands.shape}, time={time.time()-t0} sec", print=True )
        gt = [ float(sval) for sval in input_bands.spatial_ref.GeoTransform.split() ]
        input_bands.attrs['transform'] = [ gt[1], gt[2], gt[0], gt[4], gt[5], gt[3] ]
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


