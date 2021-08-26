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

    tile_size = tl.Int(1000).tag(config=True, sync=True)
    tile_index = tl.List(tl.Int, (0, 0), 2, 2).tag(config=True, sync=True)
    block_size = tl.Int(250).tag(config=True, sync=True)
    block_index = tl.List(tl.Int, (0, 0), 2, 2).tag(config=True, sync=True)
    mask_class = tl.Int(0).tag(config=True, sync=True)
    image_attrs = {}

    def __init__(self):
        super(TileManager, self).__init__()
        self._tiles: Dict[List, Tile] = {}
        self.cacheTileData = True
        self.tile_dims = None
        self.tile_shape = [ self.tile_size ] * 2
        self.block_shape = [ self.block_size ] * 2
        self.block_dims = [ self.tile_size//self.block_size ] * 2
        self._tile_data = None

    @property
    def image_name(self):
        return DataManager.instance().modal.image_name

    @property
    def iy(self):
        return self.tile_index[0]

    @property
    def ix(self):
        return self.tile_index[1]

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
        return self._tiles.setdefault(tuple(self.tile_index), Tile())

    def getTileBounds(self ) -> Tuple[ Tuple[int ,int], Tuple[int ,int] ]:
        y0, x0 = self.iy *self.tile_shape[0], self.ix *self.tile_shape[1]
        return ( y0, y0 +self.tile_shape[0] ), ( x0, x0 +self.tile_shape[1] )

    def set_tile_data_attributes(self, data: xa.DataArray):
        tr0 = data.transform
        iy0, ix0 =  self.tile_index[0] * self.tile_shape[0], self.tile_index[1] * self.tile_shape[1]
        y0, x0 = tr0[5] + iy0 * tr0[4], tr0[2] + ix0 * tr0[0]
        data.attrs['transform'] = [ tr0[0], tr0[1], x0, tr0[3], tr0[4], y0  ]
        data.attrs['tile_coords'] = self.tile_index

    def getTileFileName(self, with_extension = True ) -> str:
        tile_file_name = f"{self.image_name}.{self.fmt(self.tile_shape)}_{self.fmt(self.tile_index)}"
        return tile_file_name + ".tif" if with_extension else tile_file_name

    def tileName( self, base_name: str = None ) -> str:
        base = self.image_name if base_name is None else base_name
        return f"{base}.{self.fmt(self.tile_shape)}_{self.fmt(self.tile_index)}"

    def fmt(self, value) -> str:
        return str(value).strip("([])").replace(",", "-").replace(" ", "")

    def setTilesPerImage( self, image_specs = None ):
        ishape = image_specs['shape'] if image_specs else [ self.tile_size, self.tile_size ]
        self.tile_dims = get_rounded_dims( ishape, [self.tile_size ] *2 )
        self.tile_shape = get_rounded_dims( ishape, self.tile_dims )
        self.block_dims = get_rounded_dims( self.tile_shape, [ self.block_size ]* 2)
        self.block_shape = get_rounded_dims(self.tile_shape, self.block_dims)

    def getTileData(self) -> xa.DataArray:
        if self._tile_data is None:
            try:                    tile_data: xa.DataArray = self._readTileFile()
            except AssertionError:  tile_data = self._getTileDataFromImage()
            tile_data = self.mask_nodata(tile_data)
            init_shape = [*tile_data.shape]
            valid_bands = DataManager.instance().valid_bands()
            if valid_bands is not None:
                dataslices = [tile_data.isel(band=slice(valid_band[0], valid_band[1])) for valid_band in valid_bands]
                tile_data = xa.concat(dataslices, dim="band")
                lgm().log( f"-------------\n         ***** Selecting valid bands ({valid_bands}), init_shape = {init_shape}, resulting Tile shape = {tile_data.shape}")
            result = self.rescale(tile_data)
            self._tile_data = result
        return self._tile_data

    def getPointData( self ) -> Tuple[xa.DataArray,xa.DataArray]:
        from spectraclass.data.spatial.manager import SpatialDataManager
        tile_data: xa.DataArray = self.getTileData()
        result: xa.DataArray =  SpatialDataManager.raster2points( tile_data )
        point_coords: xa.DataArray = result.samples
        point_data = result.assign_coords( samples = np.arange( 0, point_coords.shape[0] ) )
#        samples_axis = spectra.coords['samples']
        point_data.attrs['type'] = 'tile'
        point_data.attrs['dsid'] = result.attrs['dsid']
        return ( point_data, point_coords)

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
        lgm().log( f"Tile transform: {tr0}, Block transform: {tr1}, tile indices = [{self.tile_index}], block indices = [ {iy}, {ix} ]" )
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

    def _getTileDataFromImage(self) -> xa.DataArray:
        tm = TileManager.instance()
        tm.setTilesPerImage()
        full_input_bands: xa.DataArray = DataManager.instance().modal.readSpectralData(False)
        ybounds, xbounds = tm.getTileBounds()
        if full_input_bands.attrs['fileformat'] == 'mat':
            tsize_test = ( ybounds[1] >= full_input_bands.shape[1] ) and ( xbounds[1] >= full_input_bands.shape[2] )
            assert tsize_test, "For matlab files the tile size and block size must be greater then the data's spatial dimensions"
        tile_raster = full_input_bands[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        tile_raster.attrs['tilename'] = tm.tileName()
        tile_raster.attrs['image'] = self.image_name
        tile_raster.attrs['image_shape'] = full_input_bands.shape
        self.image_attrs[self.image_name] = dict(shape=full_input_bands.shape[-2:], attrs=full_input_bands.attrs)
        tm.set_tile_data_attributes(tile_raster)
        if self.cacheTileData: DataManager.instance().modal.writeGeotiff( tile_raster )
        return tile_raster

    def _readTileFile(self) -> xa.DataArray:
        assert self.cacheTileData, "Tile file not cached."
        image_specs = self.image_attrs.get(self.image_name, None)
        TileManager.instance().setTilesPerImage(image_specs)
        tile_raster: xa.DataArray = DataManager.instance().modal.readSpectralData(True)
        if tile_raster is not None:
            tile_raster.name = self.tileName()
            tile_raster.attrs['tilename'] = self.tileName()
        return tile_raster

    @classmethod
    def mask_nodata(self, raster: xa.DataArray) -> xa.DataArray:
        nodata_value = raster.attrs.get('data_ignore_value', -9999)
        return raster.where(raster != nodata_value, float('nan'))