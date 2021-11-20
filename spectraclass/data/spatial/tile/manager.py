from skimage.transform import ProjectiveTransform
import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
from pyproj import Proj, transform
from spectraclass.data.base import DataManager, DataType
from spectraclass.util.logs import LogManager, lgm
import os, math, pickle
import traitlets.config as tlc
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable, Marker
from .tile import Tile, Block

def get_rounded_dims( master_shape: List[int], subset_shape: List[int] ) -> List[int]:
    dims = [ int(round(ms/ss)) for (ms,ss) in zip(master_shape,subset_shape) ]
    return [ max(d, 1) for d in dims ]

def tm() -> "TileManager":
    return TileManager.instance()

class TileManager(SCSingletonConfigurable):

    block_size = tl.Int(250).tag(config=True, sync=True)
    block_index = tl.List( tl.Int, (0, 0), 2, 2).tag(config=True, sync=True)
    mask_class = tl.Int(0).tag(config=True, sync=True)
    image_attrs = {}

    def __init__(self):
        super(TileManager, self).__init__()
        self._tiles: Dict[Tuple,Tile] = {}
        self.cacheTileData = True
        self._tile_size = None
        self._block_dims = None
        self.block_shape = [ self.block_size ] * 2
        self._tile_data: xa.DataArray = None
        self.crs = "+a=6378137.0 +b=6378137.0 +nadgrids=@null +proj=merc +lon_0=0.0 +x_0=0.0 +y_0=0.0 +units=m +no_defs"

    @property
    def block_dims(self) -> Tuple[int,int]:
        if self._block_dims is None:
            idata: xa.DataArray = self.getTileData()
            image_shape = idata.shape if (idata.ndim == 2) else  idata.shape[1:]
            self._block_dims = [ math.ceil(image_shape[i]/self.block_shape[i]) for i in (0,1) ]
        return self._block_dims

    @property
    def tile_size(self) -> Tuple[int,int]:
        if self._tile_size is None:
            self._tile_size = [ (self._block_dims[i] * self.block_shape[i]) for i in (0,1) ]
        return self._tile_size

    @property
    def image_name(self):
        return DataManager.instance().modal.image_name

    def getBlock(self) -> Block:
        return self.tile.getBlock( self.block_index[0], self.block_index[1] )

    def getMask(self) -> Optional[xa.DataArray]:
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        if self.mask_class < 1: return None
        mask = None
        mvar = f"mask-{self.mask_class}"
        mask_file = dm().mask_file
        if os.path.exists( mask_file ):
            mask_dset: xa.Dataset = xa.open_dataset( mask_file )
            if mvar in mask_dset.variables:
                mask = mask_dset[mvar]
        if mask is None:
            ufm().show( f"The mask for class {self.mask_class} has not yet been generated.", "red")
            lgm().log( f"Can't apply mask for class {self.mask_class} because it has not yet been generated. Mask file: {mask_file}" )
        return mask


    def get_marker(self, lon: float, lat: float, cid: int =-1, **kwargs ) -> Marker:
        from spectraclass.model.labels import LabelsManager, lm
        block = self.getBlock()
        proj = Proj( block.data.attrs.get( 'wkt', block.data.spatial_ref.crs_wkt ) )
        x, y = proj( lon, lat )
        pid = block.coords2pindex( y, x )
        assert pid >= 0, f"Marker selection error, no points for coord: {[y, x]}"
        ic = cid if (cid >= 0) else lm().current_cid
        return Marker( [pid], ic, **kwargs )

    @property
    def tile(self) -> Tile:
        return self._tiles.setdefault( self.image_name, Tile() )

    def getTileFileName(self, with_extension = True ) -> str:
        return self.image_name + ".tif" if with_extension else self.image_name

    def tileName( self, base_name: str = None ) -> str:
        base = self.image_name if base_name is None else base_name
        return base

    def fmt(self, value) -> str:
        return str(value).strip("([])").replace(",", "-").replace(" ", "")

    def getTileData(self) -> xa.DataArray:
        if self._tile_data is None:
            tile_data: xa.DataArray = self._readTileFile()
            tile_data = self.mask_nodata(tile_data)
            init_shape = [*tile_data.shape]
            valid_bands = DataManager.instance().valid_bands()
            if valid_bands is not None:
                band_names = tile_data.attrs.get( 'long_name', None )
                dataslices = [ tile_data.isel(band=slice(valid_band[0], valid_band[1])) for valid_band in valid_bands ]
                tile_data = xa.concat(dataslices, dim="band")
                if isinstance(band_names, (list, tuple)):
                    tile_data.attrs['long_name'] = sum( [ list(band_names[valid_band[0]:valid_band[1]]) for valid_band in valid_bands ], [] )
                lgm().log( f"-------------\n         ***** Selecting valid bands ({valid_bands}), init_shape = {init_shape}, resulting Tile shape = {tile_data.shape}")
            result = self.rescale(tile_data).rio.reproject(self.crs)
            result.attrs['wkt'] = result.spatial_ref.crs_wkt
            gt = [ float(tv) for tv in result.spatial_ref.GeoTransform.split() ]
            result.attrs['wkt'] = result.spatial_ref.crs_wkt
            result.attrs['transform'] = [ gt[1], gt[2], gt[0], gt[4], gt[5], gt[3], 0.0, 0.0, 1.0 ]
            result.attrs['long_name'] = tile_data.attrs.get( 'long_name', None )
            lgm().log( f" BLOCK attrs: {result.attrs}" )
            self._tile_data = result
            self.saveMetadata( )
        return self._tile_data

    def saveMetadata(self ):
        file_path = DataManager.instance().modal.getMetadataFilePath()
        print( f"Writing metadata file: {file_path}")
        with open( file_path, "w" ) as mdfile:
            mdfile.write( f"tile_shape={self._tile_data.shape}\n")
            for aid,aiv in self._tile_data.attrs.items():
                mdfile.write(f"{aid}={aiv}\n")

#     def getPointData( self ) -> Tuple[xa.DataArray,xa.DataArray]:
#         from spectraclass.data.spatial.manager import SpatialDataManager
#         tile_data: xa.DataArray = self.getTileData()
#         result: xa.DataArray =  SpatialDataManager.raster2points( tile_data )
#         point_coords: xa.DataArray = result.samples
#         point_data = result.assign_coords( samples = np.arange( 0, point_coords.shape[0] ) )
# #        samples_axis = spectra.coords['samples']
#         point_data.attrs['type'] = 'tile'
#         point_data.attrs['dsid'] = result.attrs['dsid']
#         return ( point_data, point_coords)

    def rescale(self, raster: xa.DataArray, **kwargs ) -> xa.DataArray:
        norm_type = kwargs.get( 'norm', 'spectral' )
        refresh = kwargs.get('refresh', False )
        if norm_type == "none":
            result = raster
        else:
            if norm_type == "spatial":
                norm: xa.DataArray = self._computeSpatialNorm( raster, refresh )
            else:          # 'spectral'
                norm: xa.DataArray = raster.mean( dim=['band'], skipna=True )
            result =  raster / norm
            result.attrs = raster.attrs
        return result

    @property
    def normFileName( self ) -> str:
        return f"global_norm.pkl"

    def get_block_transform( self, iy, ix ) -> ProjectiveTransform:
        tr0 = self.tile.data.attrs['transform']
        iy0, ix0 = iy * self.block_shape[0], ix * self.block_shape[1]
        y0, x0 = tr0[5] + iy0 * tr0[4], tr0[2] + ix0 * tr0[0]
        tr1 = [ tr0[0], tr0[1], x0, tr0[3], tr0[4], y0, 0, 0, 1  ]
        lgm().log( f"Tile transform: {tr0}, Block transform: {tr1}, block indices = [ {iy}, {ix} ]" )
        return  ProjectiveTransform( np.array(tr1).reshape(3, 3) )

    def _computeSpatialNorm(self, tile_raster: xa.DataArray, refresh=False) -> xa.DataArray:
        norm_file = os.path.join(self.data_cache, self.normFileName)
        if not refresh and os.path.isfile(norm_file):
            lgm().log(f"Loading norm from global norm file {norm_file}")
            return xa.DataArray.from_dict(pickle.load(open(norm_file, 'rb')))
        else:
            lgm().log(f"Computing norm and saving to global norm file {norm_file}")
            norm: xa.DataArray = tile_raster.mean(dim=['x', 'y'], skipna=True)
            pickle.dump(norm.to_dict(), open(norm_file, 'wb'))
            return norm

    def _readTileFile(self) -> xa.DataArray:
        tm = TileManager.instance()
        tile_raster: xa.DataArray = DataManager.instance().modal.readSpectralData()
        if tile_raster is not None:
            tile_raster.name = self.tileName()
            tile_raster.attrs['tilename'] = tm.tileName()
            tile_raster.attrs['image'] = self.image_name
            tile_raster.attrs['image_shape'] = tile_raster.shape
            self.image_attrs[self.image_name] = dict( shape=tile_raster.shape[-2:], attrs=tile_raster.attrs )
        return tile_raster

    @classmethod
    def mask_nodata(self, raster: xa.DataArray) -> xa.DataArray:
        nodata_value = raster.attrs.get('data_ignore_value', -9999)
        return raster.where(raster != nodata_value, float('nan'))