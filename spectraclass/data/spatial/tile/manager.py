from skimage.transform import ProjectiveTransform
import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
from pyproj import Proj, transform
from spectraclass.data.base import DataManager, DataType
import rioxarray as rio
import os, math, pickle
import traitlets.config as tlc
import traitlets as tl
from spectraclass.model.base import SCConfigurable
from .tile import Tile, Block

def get_rounded_dims( master_shape: List[int], subset_shape: List[int] ) -> List[int]:
    dims = [ int(round(ms/ss)) for (ms,ss) in zip(master_shape,subset_shape) ]
    return [ max(d, 1) for d in dims ]

class TileManager(tlc.SingletonConfigurable, SCConfigurable):

    tile_size = tl.Int(1000).tag(config=True, sync=True)
    tile_index = tl.List(tl.Int, (0, 0), 2, 2).tag(config=True, sync=True)
    tile_shape = tl.List(tl.Int ,(1000 ,1000) ,2 ,2).tag(config=True, sync=True)
    tile_dims = tl.List(tl.Int ,(4 ,4) ,2 ,2).tag(config=True, sync=True)

    block_size = tl.Int(250).tag(config=True, sync=True)
    block_shape = tl.List(tl.Int ,(250 ,250) ,2 ,2).tag(config=True, sync=True)
    block_dims = tl.List(tl.Int ,(4 ,4) ,2 ,2).tag(config=True, sync=True)
    block_index = tl.List(tl.Int, (0, 0), 2, 2).tag(config=True, sync=True)

    image_name = tl.Unicode("NONE").tag(config=True ,sync=True)
    data_cache = tl.Unicode("NONE").tag(config=True, sync=True)
    data_dir = tl.Unicode("NONE").tag(config=True, sync=True)
    image_attrs = {}

    def __init__(self):
        super(TileManager, self).__init__()
        self._tiles: Dict[List, Tile] = {}
        self.cacheTileData = True

    def tileFileName(self) -> str:
        return self.getTileFileName( self.image_name )

    @property
    def config_mode(self):
        return DataManager.instance().mode

    @property
    def iy(self):
        return self.tile_index[0]

    @property
    def ix(self):
        return self.tile_index[1]

    def getBlock(self) -> Block:
        return self.tile.getBlock( self.block_index[0], self.block_index[1] )

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

    def getTileFileName(self, image_name: str ) -> str:
        return f"{image_name}.{self._fmt(self.tile_shape)}_{self._fmt(self.tile_index)}"

    def _fmt(self, value) -> str:
        return str(value).strip("([])").replace(",", "-").replace(" ", "")

    def setTilesPerImage( self, image_specs ):
        ishape = image_specs['shape'] if image_specs else [ self.tile_size, self.tile_size ]
        self.tile_dims = get_rounded_dims( ishape, [self.tile_size ] *2 )
        self.tile_shape = get_rounded_dims( ishape, self.tile_dims )
        self.block_dims = get_rounded_dims( self.tile_shape, [ self.block_size ]* 2)
        self.block_shape = get_rounded_dims(self.tile_shape, self.block_dims)

    def getTileData(self) -> Optional[xa.DataArray]:
        tile_data: Optional[xa.DataArray] = self._readTileFile() if self.cacheTileData else None
        if tile_data is None: tile_data = self._getTileDataFromImage()
        if tile_data is None: return None
        tile_data = self.mask_nodata(tile_data)
        init_shape = [*tile_data.shape]
        valid_bands = [[0, 193], [214, 283], [319,
                                              421]]  # self.config.value('data/valid_bands', None ) # [[0, 195], [214, 286], [319, 421]] #
        if valid_bands is not None:
            dataslices = [tile_data.isel(band=slice(valid_band[0], valid_band[1])) for valid_band in valid_bands]
            tile_data = xa.concat(dataslices, dim="band")
            print(
                f"-------------\n         ***** Selecting valid bands ({valid_bands}), init_shape = {init_shape}, resulting Tile shape = {tile_data.shape}")
        result = self.rescale(tile_data)
        return result

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
        print( f"Tile transform: {tr0}, Block transform: {tr1}, tile indices = [{self.tile_index}], block indices = [ {iy}, {ix} ]" )
        return  ProjectiveTransform( np.array(tr1).reshape(3, 3) )

    def _computeSpatialNorm(self, tile_raster: xa.DataArray, refresh=False) -> xa.DataArray:
        norm_file = os.path.join(self.data_cache, self.normFileName)
        if not refresh and os.path.isfile(norm_file):
            print(f"Loading norm from global norm file {norm_file}")
            return xa.DataArray.from_dict(pickle.load(open(norm_file, 'rb')))
        else:
            print(f"Computing norm and saving to global norm file {norm_file}")
            norm: xa.DataArray = tile_raster.mean(dim=['x', 'y'], skipna=True)
            pickle.dump(norm.to_dict(), open(norm_file, 'wb'))
            return norm

    def _getTileDataFromImage(self) -> Optional[xa.DataArray]:
        tm = TileManager.instance()
        full_input_bands: xa.DataArray = self.readGeotiff(self.image_name)
        if full_input_bands is None: return None
        image_attrs = dict(shape=full_input_bands.shape[-2:], attrs=full_input_bands.attrs)
        tm.setTilesPerImage(image_attrs)
        ybounds, xbounds = tm.getTileBounds()
        tile_raster = full_input_bands[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        tile_filename = tm.tileFileName()
        tile_raster.attrs['filename'] = tile_filename
        tile_raster.attrs['image'] = self.image_name
        tile_raster.attrs['image_shape'] = full_input_bands.shape
        self.image_attrs[self.image_name] = image_attrs
        tm.set_tile_data_attributes(tile_raster)
        if self.cacheTileData: self.writeGeotiff(tile_raster, tile_filename)
        return tile_raster

    def _readTileFile(self, iband=-1) -> Optional[xa.DataArray]:
        tile_filename = self.tileFileName()
        print(f"Reading tile file {tile_filename}")
        tile_raster: Optional[xa.DataArray] = self.readGeotiff(tile_filename, iband)
        if tile_raster is not None:
            tile_raster.name = f"{self.image_name}: Band {iband + 1}" if (iband >= 0) else self.image_name
            tile_raster.attrs['filename'] = tile_filename
            image_specs = self.image_attrs.get(self.image_name, None)
            TileManager.instance().setTilesPerImage(image_specs)
        return tile_raster

    def writeGeotiff(self, raster_data: xa.DataArray, filename: str = None) -> Optional[str]:
        if filename is None: filename = raster_data.name
        if not filename.endswith(".tif"): filename = filename + ".tif"
        output_file = os.path.join(self.data_cache, filename)
        try:
            print(f"Writing (raster) tile file {output_file}")
            raster_data.rio.to_raster(output_file)
            return output_file
        except Exception as err:
            print(f"Unable to write raster file to {output_file}: {err}")
            return None

    def readGeotiff(self, filename: str, iband=-1) -> Optional[xa.DataArray]:
        if not filename.endswith(".tif"): filename = filename + ".tif"
        try:
            input_file = os.path.join(self.data_dir, filename)
            input_bands: xa.DataArray = rio.open_rasterio(input_file)
            if 'transform' not in input_bands.attrs.keys():
                gts = input_bands.spatial_ref.GeoTransform.split()
                input_bands.attrs['transform'] = [float(gts[i]) for i in [1, 2, 0, 4, 5, 3]]
            print(f"Reading raster file {input_file}, dims = {input_bands.dims}, shape = {input_bands.shape}")
            if iband >= 0:
                return input_bands[iband]
            else:
                return input_bands
        except Exception as err:
            print(f"WARNING: can't read input file {filename}: {err}")
            return None

    @classmethod
    def mask_nodata(self, raster: xa.DataArray) -> xa.DataArray:
        nodata_value = raster.attrs.get('data_ignore_value', -9999)
        return raster.where(raster != nodata_value, float('nan'))