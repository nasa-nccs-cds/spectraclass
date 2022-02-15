from skimage.transform import ProjectiveTransform
import numpy as np
import numpy.ma as ma
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from pyproj import Transformer
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
import os, math, pickle, time

class DataContainer:

    def __init__(self, **kwargs):
        super(DataContainer, self).__init__()
        self._data_projected = kwargs.get( 'data_projected', False )
        self._extent: List[float] = None
        self._transform: List[float] = None
        self._ptransform: ProjectiveTransform = None
        self._data: Optional[xa.DataArray] = None
        self._transformer: Transformer = None
        self._xlim = None
        self._ylim = None

    def reset(self):
        self._data = None

    @property
    def transformer(self) -> Transformer:
        from spectraclass.data.spatial.tile.manager import TileManager
        if self._transformer is None:
            self._transformer = Transformer.from_crs( self.data.spatial_ref.crs_wkt, f'epsg:{TileManager.ESPG}' )
        return self._transformer

    @property
    def data(self) -> xa.DataArray:
        if self._data is None:
            self._data = self._get_data()
        return self._data

    def _get_data(self) -> xa.DataArray:
        raise NotImplementedError(f" Attempt to call abstract method _get_data on {self.__class__.__name__}")

    def update_transform(self):
#        self.transformer.transform_bounds()
#        gt = self.data.attrs['transform']
        (xr, yr) = ( self.xrange, self.yrange ) if self._data_projected else self.transformer.transform( self.xrange, self.yrange  )
        dx, dy = (xr[1]-xr[0])/(self.data.shape[-1]-1), (yr[1]-yr[0])/(self.data.shape[-2]-1)
        self._extent = [ xr[0]-dx/2,  xr[-1]+dx/2,  yr[-1]-dy/2,  yr[0]+dy/2 ]
        self._transform = ( dx, 0, xr[0], 0, dy, yr[0] )

    @property
    def extent(self) -> List[float]:
        if self._extent is None: self.update_transform()
        return self._extent

    @property
    def transform(self) -> List[float]:
        if self._transform is None: self.update_transform()
        return self._transform

    @property
    def xcoord(self) -> np.ndarray:
        return self.data.coords[self.data.dims[2]].values

    @property
    def ycoord(self) -> np.ndarray:
        return self.data.coords[self.data.dims[1]].values

    @property
    def xrange(self) -> List[float]:
        return [ self.xcoord[0], self.xcoord[-1] ]

    @property
    def yrange(self) -> List[float]:
        return [ self.ycoord[0], self.ycoord[-1] ]

    @property
    def xlim(self) -> List[float]:
        if self._xlim is None:
            xc: np.ndarray = self.xcoord
            dx = (xc[-1]-xc[0])/(xc.size-1)
            self._xlim = ( xc[0]-dx, xc[-1]+dx )
        return self._xlim

    @property
    def ylim(self) -> List[float]:
        if self._ylim is None:
            yc: np.ndarray = self.ycoord
            dy = (yc[-1]-yc[0])/(yc.size-1)
            self._ylim = ( yc[0]-dy, yc[-1]+dy )
        return self._ylim

    def inBounds(self, yc: float, xc: float ) -> bool:
        if (yc < min(self._ylim)) or (yc > max(self._ylim)): return False
        if (xc < self._xlim[0]) or (xc > self._xlim[1]): return False
        return True

    @property
    def ptransform( self ) -> ProjectiveTransform:
        if self._ptransform is None:
            projection = np.array( list(self.transform) + [ 0, 0, 1 ] ).reshape(3, 3)
            self._ptransform = ProjectiveTransform( projection )
        return self._ptransform

class Tile(DataContainer):

    def __init__(self, **kwargs ):
        super(Tile, self).__init__(**kwargs)
        self._blocks = {}
        self.subsampling: int =  kwargs.get('subsample',1)

    def _get_data(self) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager
        return TileManager.instance().getTileData()

    @property
    def name(self) -> str:
        return self.data.attrs['tilename']

    def getBlock(self, ix: int, iy: int, **kwargs ) -> Optional["Block"]:
        if (ix,iy) in self._blocks: return self._blocks[ (ix,iy) ]
        return self._blocks.setdefault( (ix,iy), Block( self, ix, iy, **kwargs ) )

    def getBlocks(self, **kwargs ) -> List["Block"]:
        from spectraclass.data.spatial.tile.manager import TileManager
        tm = TileManager.instance()
        return [ self.getBlock( ix, iy, **kwargs ) for ix in range(0,tm.block_dims[0]) for iy in range(0,tm.block_dims[1]) ]

class Block(DataContainer):

    def __init__(self, tile: Tile, ix: int, iy: int, **kwargs ):
        super(Block, self).__init__( data_projected=True, **kwargs )
        self.tile: Tile = tile
        self.init_task = None
        self.config = kwargs
        self.tmask = None
        self.block_coords = (ix,iy)
        self.validate_parameters()
        self._index_array: xa.DataArray = None
        self._pid_array: np.ndarray = None
        self._flow = None
        self._samples_axis: Optional[xa.DataArray] = None
        self._point_data: Optional[xa.DataArray] = None
        self._point_coords: Dict[str,np.ndarray] = None
        lgm().log(f"CREATE Block: ix={ix}, iy={iy}")

    def classmap(self, default_value: int =0 ) -> xa.DataArray:
        return xa.full_like( self.data[0].squeeze(drop=True), default_value, dtype=np.int )

    @property
    def index_array(self):
        if self._index_array is None:
            self._index_array = self.get_index_array()
        return self._index_array

    @property
    def pid_array(self):
        if self._pid_array is None:
            self._pid_array = self.get_pid_array()
        return self._pid_array

    def dsid( self ):
        from spectraclass.data.spatial.tile.manager import TileManager
        return "-".join( [ TileManager.instance().tileName() ] + [ str(i) for i in self.block_coords ] )

    def validate_parameters(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        assert ( self.block_coords[0] < tm().block_dims[0] ) and ( self.block_coords[1] < tm().block_dims[1] ), f"Block coordinates {self.block_coords} out of bounds with block dims = {tm().block_dims}"

    @log_timing
    def _get_data( self ) -> xa.DataArray:
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        block_data_file = dm().modal.dataFile( block=self )
        if os.path.isfile( block_data_file ):
            dataset: Optional[xa.Dataset] = dm().modal.loadDataFile(block=self)
            raw_raster = dataset["raw"]
            if raw_raster.size == 0: ufm().show( "This block does not appear to have any data.", "red" )
        else:
            if self.tile.data is None: return None
            xbounds, ybounds = self.getBounds()
            raster_slice = self.tile.data[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
            raw_raster = TileManager.process_tile_data( raster_slice )
        block_raster = self._apply_mask( raw_raster )
        block_raster.attrs['block_coords'] = self.block_coords
        block_raster.attrs['dsid'] = self.dsid()
        block_raster.attrs['file_name'] = self.file_name
        block_raster.name = self.file_name
        return block_raster

    def _apply_mask(self, block_array: xa.DataArray ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        nodata_value = np.nan
        mask_array: Optional[xa.DataArray] = tm().getMask()
        return block_array if mask_array is None else block_array.where( mask_array, nodata_value )

    def addTextureBands( self ):
        from spectraclass.features.texture.manager import TextureManager, texm
        self._data = texm().addTextureBands( self.data )

    @property
    def file_name(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return f"{tm().tileName()}_b-{tm().fmt(self.shape)}-{self.block_coords[0]}-{self.block_coords[1]}"

    def get_index_array(self) -> xa.DataArray:
        stacked_data: xa.DataArray = self.data.stack( samples=self.data.dims[-2:] )
        filtered_samples = stacked_data[1].dropna( dim="samples" )
        indices = np.arange(filtered_samples.shape[0])
        point_indices = xa.DataArray( indices, dims=['samples'], coords=dict(samples=filtered_samples.samples) )
        result = point_indices.reindex( samples=stacked_data.samples, fill_value= -1 )
        return result.unstack()

    def get_pid_array(self) -> np.ndarray:
        d0: np.ndarray = self.data.values[0].squeeze().flatten()
        pids = np.arange(d0.size)
        return pids[ ~np.isnan(d0) ]

    @property
    def shape(self) -> Tuple[int,int]:
        from spectraclass.data.spatial.tile.manager import TileManager
        return TileManager.instance().block_shape

    @property
    def zeros(self) -> np.ndarray:
        return np.zeros( self.shape, np.int)

    def getBounds(self ) -> Tuple[ Tuple[int,int], Tuple[int,int] ]:
        x0, y0 = self.block_coords[0]*self.shape[0], self.block_coords[1]*self.shape[1]
        return ( x0, x0+self.shape[1] ), ( y0, y0+self.shape[0] )

    def getPointData( self ) -> Tuple[xa.DataArray,Dict]:
        if self._point_data is None:
            result, mask =  self.raster2points( self.data )
            self._point_coords: Dict[str,np.ndarray] = dict( y=self.data.y.values, x=self.data.x.values, mask=mask )
            self._point_data = result.assign_coords( samples = np.arange( 0, result.shape[0] ) )
            self._samples_axis = self._point_data.coords['samples']
            self._point_data.attrs['type'] = 'block'
            self._point_data.attrs['dsid'] = self.dsid()
        return (self._point_data, self._point_coords )

    @property
    def point_coords(self) -> Dict[str,np.ndarray]:
        if self._point_coords is None: self.getPointData()
        return  self._point_coords

    @property
    def mask(self) -> np.ndarray:
        return self.point_coords['mask']

    def getSelectedPointData( self, cy: List[float], cx: List[float] ) -> np.ndarray:
        yIndices, xIndices = self.multi_coords2indices(cy, cx)
        return  self.data.values[ :, yIndices, xIndices ].transpose()

    def getSelectedPointIndices( self, cy: List[float], cx: List[float] ) -> np.ndarray:
        yIndices, xIndices = self.multi_coords2indices(cy, cx)
        return  yIndices * self.shape[1] + xIndices

    def getSelectedPoint( self, cy: float, cx: float ) -> np.ndarray:
        index = self.coords2indices(cy, cx)
        return self.data[ :, index['iy'], index['ix'] ].values.reshape(1, -1)

    # def plot(self,  **kwargs ) -> xa.DataArray:
    #     from spectraclass.data.spatial.manager import SpatialDataManager
    #     color_band = kwargs.pop( 'color_band', None )
    #     band_range = kwargs.pop( 'band_range', None )
    #     if color_band is not None:
    #         plot_data = self.data[color_band]
    #     elif band_range is not None:
    #         plot_data = self.data.isel( band=slice( band_range[0], band_range[1] ) ).mean(dim="band", skipna=True)
    #     else:
    #         plot_data =  SpatialDataManager.getRGB(self.data)
    #     SpatialDataManager.plotRaster( plot_data, **kwargs )
    #     return plot_data

    def coords2indices(self, cy, cx) -> Dict:
        coords = self.ptransform.inverse(np.array([[cx, cy], ]))
        return dict( iy =math.floor(coords[0, 1]), ix = math.floor(coords[0, 0]) )

    def multi_coords2indices(self, cy: List[float], cx: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.array( list( zip( cx, cy ) ) )
        trans_coords = np.floor(self.ptransform.inverse(coords))
        indices = trans_coords.transpose().astype( np.int32 )
        return indices[1], indices[0]

    def indices2coords(self, iy, ix) -> Dict:
        (iy,ix) = self.ptransform(np.array([[ix+0.5, iy+0.5], ]))
        return dict( iy = iy, ix = ix )

    def pid2coords(self, pid: int) -> Dict:
        point_index = self.pid_array[pid]
        xs = self.point_coords['x'].size
        pi = dict( x= point_index % xs,  y= point_index // xs )
        try:
            return { c: self.point_coords[c][ pi[c] ] for c in ['y','x'] }
        except Exception as err:
            lgm().log( f" --> pindex2coords Error: {err}, pid = {point_index}, coords = {pi}" )

    def pid2indices(self, pid: int) -> Dict:
        point_index = self.pid_array[pid]
        xs = self.point_coords['x'].size
        pi = dict( x= point_index % xs,  y= point_index // xs )
        try:
            selected_sample: List = [ self.point_coords[c][ pi[c] ] for c in ['y','x'] ]
            return self.coords2indices( selected_sample[0], selected_sample[1] )
        except Exception as err:
            lgm().log( f" --> pindex2indices Error: {err}, pid = {point_index}, coords = {pi}" )

    def points2raster(self, points_data: xa.DataArray ):
        lgm().log( f"points->raster, points: dims={points_data.dims}, shape={points_data.shape}")
        dims = [points_data.dims[1], self.data.dims[1], self.data.dims[2]]
        coords = [(dims[0], points_data[dims[0]].data), (dims[1], self.data[dims[1]].data), (dims[2], self.data[dims[2]].data)]
        raster_data = np.full([self.data.shape[1] * self.data.shape[2], points_data.shape[1]], float('nan'))
        raster_data[ self.mask ] = points_data.data
        raster_data = raster_data.transpose().reshape([points_data.shape[1], self.data.shape[1], self.data.shape[2]])
        lgm().log( f"Generated Raster data, shape={raster_data.shape}, dims={dims}, with mask shape={self.mask.shape}" )
        return xa.DataArray( raster_data, coords, dims, points_data.name, points_data.attrs )

    @classmethod
    def raster2points( cls, base_raster: xa.DataArray ) -> Tuple[xa.DataArray,np.ndarray]:   #  base_raster dims: [ band, y, x ]
        t0 = time.time()
        point_data = base_raster.stack(samples=base_raster.dims[-2:]).transpose()
        if '_FillValue' in point_data.attrs:
            nodata = point_data.attrs['_FillValue']
            point_data = point_data if np.isnan( nodata ) else point_data.where( point_data != nodata, np.nan )
        mask: np.ndarray = ~np.isnan(point_data.values).any(axis=1)
        filtered_point_data: xa.DataArray = point_data[ mask, : ]
        filtered_point_data.attrs['dsid'] = base_raster.name
        lgm().log( f"raster2points -> [{base_raster.name}]: filtered_point_data shape = {filtered_point_data.shape}" )
        lgm().log( f" --> mask shape = {mask.shape}, mask #valid = {np.count_nonzero(mask)}/{mask.size}, completed in {time.time()-t0} sec" )
        return filtered_point_data, mask

    def coords2pindex( self, cy, cx ) -> int:
        try:
            index = self.coords2indices( cy, cx )
            return self.index_array.values[ index['iy'], index['ix'] ]
        except IndexError as err:
            return -1

    def multi_coords2pindex(self, ycoords: List[float], xcoords: List[float] ) -> np.ndarray:
        ( yi, xi ) = self.multi_coords2indices( ycoords, xcoords )
        return self.index_array.values[ yi, xi ]


