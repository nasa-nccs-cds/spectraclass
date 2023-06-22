import numpy as np
import codecs
import xarray as xa
import geoviews.tile_sources as gts
import holoviews as hv
import shapely.vectorized as svect
from typing import List, Union, Tuple, Optional, Dict
from pyproj import Proj
from spectraclass.util.logs import LogManager, lgm, log_timing
import os, param, math, pickle, time
import cartopy.crs as ccrs
from spectraclass.util.logs import lgm, exception_handled
from spectraclass.widgets.polygon import PolyRec
import traitlets as tl
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.gui.spatial.widgets.markers import Marker
import geoviews as gv
from pyproj import Transformer
from .tile import Tile, Block
from spectraclass.gui.control import UserFeedbackManager, ufm

def get_rounded_dims( master_shape: List[int], subset_shape: List[int] ) -> List[int]:
    dims = [ int(round(ms/ss)) for (ms,ss) in zip(master_shape,subset_shape) ]
    return [ max(d, 1) for d in dims ]

def nnan( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return np.count_nonzero( np.isnan(array) )

def tm() -> "TileManager":
    return TileManager.instance()

class PointsOutOfBoundsException(Exception):
    def __str__(self):
        return "Points out of bounds"

class BlockSelection(param.Parameterized):
    index = param.Integer(default=-1, doc="selected block index")

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
        self._scale: Tuple[np.ndarray,np.ndarray] = None
        self.block_selection = BlockSelection()

    def bi2c(self, bindex: int ) -> Tuple[int,int]:
        ts1: int = self.tile_shape[1]
        return ( bindex//ts1, bindex%ts1 )

    def c2bi(self, bcoords: Tuple[int,int] ) -> int:
        ts1 = self.tile_shape[1]
        return bcoords[0]*ts1 + bcoords[1]

    def getESRIImageryServer(self,**kwargs) -> gv.element.geo.Tiles:
        url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.jpg'
        lgm().log( f"TM: getESRIImageryServer{kwargs} ")
        return gv.element.geo.WMTS( url, name="EsriImagery").opts( **kwargs )

    @classmethod
    def encode( cls, obj ) -> str:
        return codecs.encode(pickle.dumps(obj), "base64").decode()

    @classmethod
    def decode( cls, pickled: str ):
        if pickled: return pickle.loads(codecs.decode(pickled.encode(), "base64"))

    @property
    def block_shape(self):
        block = self.getBlock( block_coords=(0,0) )
        return block.shape

    # @tl.observe('block_index')
    # def _block_index_changed(self, change):
    #     from spectraclass.gui.pointcloud import PointCloudManager, pcm
    #     pcm().refresh()

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

    def get_satellite_image(self):
        projection = ccrs.GOOGLE_MERCATOR
        block = self.getBlock()
        (xlim, ylim) = block.get_extent(projection)
        tile_source = gts.tile_sources.get("EsriImagery", None).opts(xlim=xlim, ylim=ylim, width=600, height=570)
        return tile_source

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
        ts = self.tile_shape
        return math.ceil(ts[0]/self.block_size), math.ceil(ts[1]/self.block_size)

    @property
    def tile_size(self) -> Tuple[int,int]:
        bd = self.block_dims
        return bd[0]*self.block_size, bd[1]*self.block_size

    @property
    def tile_shape(self) -> Tuple[int,int]:
        return ( self.tile.data.shape[-1], self.tile.data.shape[-2] )

    @property
    def image_name(self):
        from spectraclass.data.base import DataManager, dm, DataType
        return dm().modal.image_name

    def get_image_name( self, **kwargs ):
        from spectraclass.data.base import DataManager, dm, DataType
        image_index = kwargs.get('index', DataManager.instance().modal._active_image )
        dm().modal.set_current_image(image_index)
        return dm().modal.image_name

    @property
    def image_index(self) -> int:
        from spectraclass.data.base import DataManager, dm, DataType
        return dm().modal.image_index

    @property
    def block_coords(self) -> Tuple:
        return tuple(self.block_index)

    def setBlock( self, block_index ) -> bool:
        from spectraclass.data.base import DataManager, dm, DataType
        if tuple(block_index) != self.block_index:
            lgm().log( f"TileManager.setBlock -> {block_index}")
            ufm().show( f"Set Block: {block_index}")
            self.block_index = tuple(block_index)
            dm().loadCurrentProject( 'setBlock', True, bindex=block_index )
            self.block_selection.index = self.c2bi(block_index)
            return True
        return False

    def set_scale(self, scale: Tuple[np.ndarray,np.ndarray] ):
        self._scale = scale

    def get_scale(self) -> Tuple[np.ndarray,np.ndarray]:
        return self._scale

    def in_bounds( self, pids: List[int] ) -> bool:
        try:
            block = self.getBlock()
            point_data: xa.DataArray = block.getPointData()[0]
            result = point_data.sel( dict(samples=pids) ).values
            return True
        except KeyError:
            return False

    @exception_handled
    def getBlock( self, **kwargs ) -> Block:
#        from spectraclass.data.base import DataManager, dm
        bindex = kwargs.get( 'bindex' )
        tindex = kwargs.get( 'tindex' )
        if (bindex is None) and ('block' in kwargs): bindex = kwargs['block'].block_coords
        block_index = self.block_index if (bindex is None) else bindex
        tile: Tile = self.tile if (tindex is None) else self.get_tile( tindex )
 #       block_index = dm().modal.get_valid_block_coords( tile.index, init_bindex )
        return tile.getDataBlock( block_index[0], block_index[1] )

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
            ufm().show( f"The mask for class {self.mask_class} has not yet been generated.", "warning")
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

    def relative_to_absolute(self, polydata: Dict[str,np.ndarray] ) -> Dict[str,np.ndarray]:
        [x0,x1,y0,y1] = self.getBlock().extent
        dx, dy = x1-x0, y1-y0
        xc, yc = polydata['x'], polydata['y'],
        return dict( x = x0 + xc*dx, y = y0 + yc*dy )

    @exception_handled
    @log_timing
    def get_region_marker(self, polydata: Dict[str,np.ndarray], cid: int = -1, **kwargs ) -> Optional[Marker]:
        from spectraclass.data.spatial.tile.tile import Block, Tile
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        from shapely.geometry import Polygon
        relative = kwargs.get( 'relative', False )
        marker = None
        if cid == -1: cid = lm().current_cid
        block: Block = self.getBlock()
        raster:  xa.DataArray = block.data[0].squeeze()
        X, Y = raster.x.values, raster.y.values
        if relative: polydata = self.relative_to_absolute( polydata )
        try:
            pdata = np.stack( [ polydata['x'], polydata['y'] ], axis=-1 )
            lgm().log(f"Poly selection-> Polygons data: {polydata}, stacked data shape: {pdata.shape}")
            polygon = Polygon( pdata )
            MX, MY = np.meshgrid(X, Y)
            PID: np.ndarray = np.array(range(raster.size))
            mask: np.ndarray = svect.contains( polygon, MX, MY ).flatten()
            mask_pids = PID[mask] # idx2pid[ PID[mask] ]
            pids = mask_pids[ mask_pids > -1 ].tolist()
            if not self.in_bounds( pids ): raise PointsOutOfBoundsException()
            marker = Marker( "label", pids, cid )
            lgm().log( f"Poly selection-> Create marker[{marker.size}], cid = {cid}")
        except Exception as err:
            lgm().exception( f"Error getting region marker, returning empty marker: {err}")
            ufm().show( str(err), "warning" )
        return marker

    @exception_handled
    @log_timing
    def get_region_marker_legacy(self, prec: PolyRec, cid: int = -1) -> Optional[Marker]:
        from spectraclass.data.spatial.tile.tile import Block, Tile
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        from shapely.geometry import Polygon
        marker = None
        if cid == -1: cid = lm().current_cid
        block: Block = self.getBlock()
        raster: xa.DataArray = block.data[0].squeeze()
        X, Y = raster.x.values, raster.y.values
        try:
            #            xy = prec.poly.get_xy()
            #            [yi,xi] = block.multi_coords2indices( xy[:,1], xy[:,0] )
            polygon = Polygon(prec.poly.get_xy())
            MX, MY = np.meshgrid(X, Y)
            PID: np.ndarray = np.array(range(raster.size))
            mask: np.ndarray = svect.contains(polygon, MX, MY).flatten()
            mask_pids = PID[mask]  # idx2pid[ PID[mask] ]
            pids = mask_pids[mask_pids > -1].tolist()
            if not self.in_bounds(pids): raise PointsOutOfBoundsException()
            marker = Marker("label", pids, cid)
            lgm().log(f"Poly selection-> Create marker[{marker.size}], cid = {cid}")
        except Exception as err:
            lgm().log(f"Error getting region marker, returning empty marker: {err}")
            ufm().show(str(err), "warning")
        return marker

    def getTileFileName(self, with_extension = True ) -> str:
        from spectraclass.data.base import DataManager, dm, DataType
        ext = dm().modal.ext
        return self.image_name + ext if with_extension else self.image_name

    def tileName( self, **kwargs ) -> str:
        return self.get_image_name( **kwargs )

    @property
    def tileid(self):
        return f"{self.tileName()}_b-{self.block_size}"

    def fmt(self, value) -> str:
        return str(value).strip("([])").replace(",", "-").replace(" ", "")

    def getTileData(self) -> xa.DataArray:
         return self._readTileFile()

    @classmethod
    def filter_invalid_data( cls, tile_data: xa.DataArray ) -> xa.DataArray:
        from spectraclass.data.base import DataManager, dm, DataType
        tile_data = cls.mask_nodata(tile_data)
        valid_bands = DataManager.instance().valid_bands()
        init_shape = [*tile_data.shape]
        if valid_bands is not None:
            band_names = tile_data.attrs.get('bands', None)
            dataslices = [tile_data.isel(band=slice(valid_band[0], valid_band[1])) for valid_band in valid_bands]
            tile_data = xa.concat(dataslices, dim="band")
            if isinstance(band_names, (list, tuple)):
                tile_data.attrs['bands'] = sum( [list(band_names[valid_band[0]:valid_band[1]]) for valid_band in valid_bands], [])
            lgm().log( f"-------------\n         ***** Selecting valid bands ({valid_bands}), init_shape = {init_shape}, resulting Tile shape = {tile_data.shape}")
        return tile_data

    def count_nbands(self) -> int:
        from spectraclass.data.base import DataManager, dm, DataType
        valid_bands = dm().valid_bands()
        nbmax = self.tile.data.shape[0]
        if valid_bands is None:
            return nbmax
        else:
            nb = 0
            for valid_band in valid_bands:
                nb += ( min(nbmax,valid_band[1]) - valid_band[0] )
            return nb

    def get_band_filter_signature(self) -> str:
        from spectraclass.data.base import DataManager, dm, DataType
        valid_bands = dm().valid_bands()
        nbmax = self.tile.data.shape[0]
        if valid_bands is None:
            return "000"
        else:
            nb = 1
            for valid_band in valid_bands:
                if valid_band[0] > 0: nb *= valid_band[0]
                nb *= min(nbmax,valid_band[1])
            return str(nb)[-4:]

    @classmethod
    def process_tile_data( cls, tile_data: xa.DataArray ) -> xa.DataArray:
#        tile_data = tile_data.xgeo.reproject(espg=cls.ESPG)
#        tile_data.attrs['wkt'] = cls.crs.to_wkt()
#        tile_data.attrs['crs'] = cls.crs.to_string()
        return cls.filter_invalid_data( tile_data )

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

    # def get_block_transform( self, iy, ix ) -> ProjectiveTransform:
    #     tr0 = self.transform
    #     iy0, ix0 = iy * self.block_shape[0], ix * self.block_shape[1]
    #     y0, x0 = tr0[5] + iy0 * tr0[4], tr0[2] + ix0 * tr0[0]
    #     tr1 = [ tr0[0], tr0[1], x0, tr0[3], tr0[4], y0, 0, 0, 1  ]
    #     lgm().log( f"Tile transform: {tr0}, Block transform: {tr1}, block indices = [ {iy}, {ix} ]" )
    #     return  ProjectiveTransform( np.array(tr1).reshape(3, 3) )

    def _readTileFile(self) -> xa.DataArray:
        from spectraclass.data.base import DataManager, dm
        tm = TileManager.instance()
        tile_raster: xa.DataArray = dm().modal.readSpectralData()
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
        return raster.where(raster != nodata_value, float('nan') )

    def norm( self, data: Optional[xa.DataArray], axis=None ) -> Optional[xa.DataArray]:
        if data is not None:
            dave, dmag = np.nanmean(data.values, keepdims=True, axis=axis), np.nanstd(data.values, keepdims=True, axis=axis)
            normed_data = (data.values - dave) / dmag
            return data.copy(data=normed_data)



