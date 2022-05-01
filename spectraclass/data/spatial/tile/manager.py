from skimage.transform import ProjectiveTransform
import numpy as np
import codecs
import xarray as xa
import shapely.vectorized as svect
from typing import List, Union, Tuple, Optional, Dict
from pyproj import Proj
from spectraclass.data.base import DataManager, dm, DataType
from spectraclass.util.logs import LogManager, lgm, log_timing
import os, math, pickle, time
import cartopy.crs as ccrs
from spectraclass.util.logs import lgm, exception_handled
from spectraclass.widgets.polygon import PolyRec
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.gui.spatial.widgets.markers import Marker
from pyproj import Transformer
from .tile import Tile, Block

def get_rounded_dims( master_shape: List[int], subset_shape: List[int] ) -> List[int]:
    dims = [ int(round(ms/ss)) for (ms,ss) in zip(master_shape,subset_shape) ]
    return [ max(d, 1) for d in dims ]

def tm() -> "TileManager":
    return TileManager.instance()

class TileManager(SCSingletonConfigurable):

    block_size = tl.Int(250).tag( config=True, sync=True )
    block_index = tl.Tuple( default_value=(0,0) ).tag( config=True, sync=True )
    mask_class = tl.Int(0).tag( config=True, sync=True )
    autoprocess = tl.Bool(True).tag( config=True, sync=True )
    reprocess = tl.Bool(False).tag( config=True, sync=True )
    image_attrs = {}
    ESPG = 3857
    crs = ccrs.epsg(ESPG) # "+a=6378137.0 +b=6378137.0 +nadgrids=@null +proj=merc +lon_0=0.0 +x_0=0.0 +y_0=0.0 +units=m +no_defs"
    geotrans = Transformer.from_crs( f'epsg:{ESPG}', f'epsg:4326' )

    def __init__(self):
        super(TileManager, self).__init__()
        self._tiles: Dict[str,Tile] = {}
        self._idxtiles: Dict[int, Tile] = {}
        self.cacheTileData = True
        self._block_dims = None
        self._tile_size = None
        self._tile_shape = None

    @classmethod
    def encode( cls, obj ) -> str:
        return codecs.encode(pickle.dumps(obj), "base64").decode()

    @classmethod
    def decode( cls, pickled: str ):
        if pickled: return pickle.loads(codecs.decode(pickled.encode(), "base64"))

    @property
    def block_shape(self):
        block = self.getBlock( bindex=(0,0) )
        return block.shape

    @tl.observe('block_index')
    def _block_index_changed(self, change):
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        pcm().refresh()

    @property
    def tile(self) -> Tile:
        if self.image_name in self._tiles: return self._tiles[self.image_name]
        new_tile = Tile( self.image_index )
        self._idxtiles[ self.image_index ] = new_tile
        return self._tiles.setdefault( self.image_name, new_tile )

    def get_tile( self, tile_index: int ):
        if tile_index in self._idxtiles: return self._idxtiles[tile_index]
        new_tile = Tile( tile_index )
        self._idxtiles[ tile_index ] = new_tile
        return self._tiles.setdefault( self.get_image_name(index=tile_index), new_tile )

    def tile_grid_offset(self, tile_index: int ) -> int:
        offset = 0
        for itile in range( tile_index ):
            offset = offset + self.get_tile( itile ).grid_size
        return offset

    @property
    def extent(self):
        return self.tile.extent

    @property
    def transform(self):
        return self.tile.transform

    # @property
    # def tile_metadata(self):
    #     if self._tile_metadata is None:
    #         self._tile_metadata = self.loadMetadata()
    #     return self._tile_metadata

    @classmethod
    def reproject_to_latlon( cls, x, y ):
        return cls.geotrans.transform(  x, y )

    @property
    def block_dims(self) -> Tuple[int,int]:
        if self._block_dims is None:
            self._block_dims = [ math.ceil(self.tile_shape[i]/self.block_size) for i in (0,1) ]
        return self._block_dims

    @property
    def tile_size(self) -> Tuple[int,int]:
        if self._tile_size is None:
            self._tile_size = [ ( self.block_dims[i] * self.block_size ) for i in (0,1) ]
        return self._tile_size

    @property
    def tile_shape(self) -> Tuple[int,int]:
        if self._tile_shape is None:
            self._tile_shape = [ self.tile.data.shape[-1], self.tile.data.shape[-2] ]
        return self._tile_shape

    @property
    def image_name(self):
        return dm().modal.image_name

    def get_image_name( self, **kwargs ):
        image_index = kwargs.get('index', DataManager.instance().modal._active_image )
        return dm().modal.image_names[ image_index ]

    @property
    def image_index(self) -> int:
        return dm().modal.image_index

    @property
    def block_coords(self) -> Tuple:
        return tuple(self.block_index)

    def setBlock( self, block_index ):
        if block_index != self.block_index:
            self.block_index = block_index
            dm().loadCurrentProject( 'setBlock', True )

    @exception_handled
    def getBlock( self, **kwargs ) -> Block:
        bindex = kwargs.get( 'bindex' )
        tindex = kwargs.get( 'tindex' )
        if (bindex is None) and ('block' in kwargs): bindex = kwargs['block'].block_coords
        if bindex is not None: self.block_index = bindex
        tile = self.tile if (tindex is None) else self.get_tile( tindex )
        return tile.getDataBlock( self.block_index[0], self.block_index[1] )

    @exception_handled
    def getMask(self) -> Optional[np.ndarray]:
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
        return mask.values if (mask is not None) else None


    def get_marker(self, lon: float, lat: float, cid: int =-1, **kwargs ) -> Marker:
        from spectraclass.model.labels import LabelsManager, lm
        block = self.getBlock()
        proj = Proj( block.data.attrs.get( 'wkt', block.data.spatial_ref.crs_wkt ) )
        x, y = proj( lon, lat )
        gid,ix,iy = block.coords2gid(y, x)
        assert gid >= 0, f"Marker selection error, no points for coord[{ix},{iy}]: {[x,y]}"
        ic = cid if (cid >= 0) else lm().current_cid
        return Marker( "marker", [gid], ic, **kwargs )

    @exception_handled
    @log_timing
    def get_region_marker(self, prec: PolyRec, cid: int = -1 ) -> Marker:
        from spectraclass.data.spatial.tile.tile import Block, Tile
        from spectraclass.model.labels import LabelsManager, lm
        from shapely.geometry import Polygon
        if cid == -1: cid = lm().current_cid
        block: Block = self.getBlock()
#        idx2pid: np.ndarray = block.index_array.values.flatten()
        raster:  xa.DataArray = block.data[0].squeeze()
        X, Y = raster.x.values, raster.y.values
        try:
            polygon = Polygon(prec.poly.get_xy())
            MX, MY = np.meshgrid(X, Y)
            PID: np.ndarray = np.array(range(raster.size))
            mask: np.ndarray = svect.contains( polygon, MX, MY ).flatten()
            pids = PID[mask] # idx2pid[ PID[mask] ]
            marker = Marker( "label", pids[ pids > -1 ].tolist(), cid )
            lgm().log( f"Poly selection-> Create marker[{marker.size}], cid = {cid}")
        except Exception as err:
            lgm().log( f"Error getting region marker, returning empty marker: {err}")
            marker = Marker( "label", [], cid )
        return marker

    def getTileFileName(self, with_extension = True ) -> str:
        return self.image_name + ".tif" if with_extension else self.image_name

    def tileName( self, **kwargs ) -> str:
        return self.get_image_name( **kwargs )

    def fmt(self, value) -> str:
        return str(value).strip("([])").replace(",", "-").replace(" ", "")

    def getTileData(self) -> xa.DataArray:
         return self._readTileFile()

    @classmethod
    def process_tile_data( cls, tile_data: xa.DataArray ) -> xa.DataArray:
        t0 = time.time()
        tile_data = cls.mask_nodata(tile_data)
        init_shape = [*tile_data.shape]
        valid_bands = DataManager.instance().valid_bands()
        if valid_bands is not None:
            band_names = tile_data.attrs.get('bands', None)
            dataslices = [tile_data.isel(band=slice(valid_band[0], valid_band[1])) for valid_band in valid_bands]
            tile_data = xa.concat(dataslices, dim="band")
            if isinstance(band_names, (list, tuple)):
                tile_data.attrs['bands'] = sum( [list(band_names[valid_band[0]:valid_band[1]]) for valid_band in valid_bands], [])
            lgm().log( f"-------------\n         ***** Selecting valid bands ({valid_bands}), init_shape = {init_shape}, resulting Tile shape = {tile_data.shape}")
        t1 = time.time()
        result = tile_data.rio.reproject(cls.crs)
        result.attrs['wkt'] = result.spatial_ref.crs_wkt
        result.attrs['long_name'] = tile_data.attrs.get('long_name', None)
        if '_FillValue' in result.attrs:
           nodata = result.attrs['_FillValue']
           result = result if np.isnan(nodata) else result.where(result != nodata, np.nan)
        lgm().log(f"Completed process_tile_data in ( {time.time()-t1:.1f}, {t1-t0:.1f} ) sec")
        return result

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

    def get_block_transform( self, iy, ix ) -> ProjectiveTransform:
        tr0 = self.transform
        iy0, ix0 = iy * self.block_shape[0], ix * self.block_shape[1]
        y0, x0 = tr0[5] + iy0 * tr0[4], tr0[2] + ix0 * tr0[0]
        tr1 = [ tr0[0], tr0[1], x0, tr0[3], tr0[4], y0, 0, 0, 1  ]
        lgm().log( f"Tile transform: {tr0}, Block transform: {tr1}, block indices = [ {iy}, {ix} ]" )
        return  ProjectiveTransform( np.array(tr1).reshape(3, 3) )

    def _readTileFile(self) -> xa.DataArray:
        from spectraclass.gui.control import UserFeedbackManager, ufm
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