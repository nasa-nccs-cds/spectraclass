from skimage.transform import ProjectiveTransform
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
from pyproj import Proj, transform
import os, math, pickle

class Tile:

    def __init__(self, **kwargs ):
        super(Tile, self).__init__()
        self._data: Optional[xa.DataArray] = None
        self._transform: Optional[ProjectiveTransform] = None
        self.subsampling: int =  kwargs.get('subsample',1)

    @property
    def data(self) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager
        if self._data is None:
            self._data: xa.DataArray = TileManager.instance().getTileData()
        return self._data

    def reset(self):
        self._data = None

    @property
    def transform(self) -> Optional[ProjectiveTransform]:
        if self.data is None: return None
        if self._transform is None:
            self._transform = ProjectiveTransform( np.array(list(self.data.attrs['transform']) + [0, 0, 1]).reshape(3, 3) )
        return self._transform

    @property
    def name(self) -> str:
        return self.data.attrs['tilename']

    def getBlock(self, iy: int, ix: int, **kwargs ) -> Optional["Block"]:
        block = Block( self, iy, ix, **kwargs )
        return block

    def getBlocks(self, **kwargs ) -> List["Block"]:
        from spectraclass.data.spatial.tile.manager import TileManager
        tm = TileManager.instance()
        return [ Block( self, iy, ix, **kwargs ) for iy in range(0,tm.block_dims[0]) for ix in range(0,tm.block_dims[1]) ]

    def coords2index(self, cy, cx ) -> Tuple[int,int]:     # -> iy, ix
        coords = self.transform.inverse(np.array([[cx, cy], ]))
        return (math.floor(coords[0, 1]), math.floor(coords[0, 0]))

#    def index2coords(self, iy, ix ) -> Tuple[float,float]:
#        return self.transform(np.array([[ix+0.5, iy+0.5], ]))

class Block:

    def __init__(self, tile: Tile, iy: int, ix: int, **kwargs ):
        self.tile: Tile = tile
        self.init_task = None
        self.config = kwargs
        self.block_coords = (iy,ix)
        self.validate_parameters()
        self._data: Optional[xa.DataArray] = None
        self._index_array: xa.DataArray = None
        self._flow = None
        self._samples_axis: Optional[xa.DataArray] = None
        self._point_data: Optional[xa.DataArray] = None
        self._point_coords: Dict = None
        self._xlim = None
        self._ylim = None

    @property
    def data(self) -> Optional[xa.DataArray]:
        if self._data is None:
            self._data = self._getData()
        return self._data

    @property
    def index_array(self):
        if self._index_array is None:
            self._index_array = self.get_index_array()
        return self._index_array

    @property
    def transform( self ):
        from spectraclass.data.spatial.tile.manager import TileManager
        tr0 = self.data.attrs.get('transform')
        if tr0 is None:
            pt: ProjectiveTransform = TileManager.instance().get_block_transform( *self.block_coords )
            self._data.attrs['transform'] = pt.params.flatten().tolist()
            return pt
        else:
            tlist: List[float] = tr0.tolist() if isinstance( tr0, np.ndarray ) else tr0
            if len( tlist ) == 6: tlist = tlist + [ 0, 0, 1 ]
            projection = np.array( tlist ).reshape(3, 3)
            return ProjectiveTransform( projection )

    def dsid( self ):
        from spectraclass.data.spatial.tile.manager import TileManager
        return "-".join( [ TileManager.instance().tileName() ] + [ str(i) for i in self.block_coords ] )

    def validate_parameters(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        assert ( self.block_coords[0] < tm().block_dims[0] ) and ( self.block_coords[1] < tm().block_dims[1] ), f"Block coordinates {self.block_coords} out of bounds with block dims = {tm().block_dims}"

    def _getData( self ) -> Optional[xa.DataArray]:
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        try:
            dataset = dm().modal.loadDataFile( block=self )
            raw_raster = dataset["raw"]
        except Exception:
            if self.tile.data is None: return None
            ybounds, xbounds = self.getBounds()
            raw_raster = self.tile.data[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
        block_raster = self._apply_mask( raw_raster )
        block_raster.attrs['block_coords'] = self.block_coords
        block_raster.attrs['dsid'] = self.dsid()
        block_raster.attrs['file_name'] = self.file_name
        pt: ProjectiveTransform = tm().get_block_transform(*self.block_coords)
        block_raster.attrs['transform'] = pt.params.flatten().tolist()
        block_raster.name = self.file_name
        return block_raster

    def _apply_mask(self, block_array: xa.DataArray ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        nodata_value = np.nan
        mask_array: Optional[xa.DataArray] = tm().getMask()
        return block_array if mask_array is None else block_array.where( mask_array, nodata_value )

    def _loadTileData(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        ybounds, xbounds = self.getBounds()
        block_raster = self.tile.data[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
        block_raster.attrs['block_coords'] = self.block_coords
        block_raster.attrs['dsid'] = self.dsid()
        block_raster.attrs['file_name'] = self.file_name
        block_raster.name = self.file_name
        pt: ProjectiveTransform = tm().get_block_transform(*self.block_coords)
        block_raster.attrs['transform'] = pt.params.flatten().tolist()
        self._data = block_raster

    def clearBlockCache(self):
        from spectraclass.data.base import DataManager, dm
        block_file = dm().modal.blockFilePath( block = self )
        if os.path.exists(block_file):
            lgm().log( f"Removing block file: {block_file} ")
            os.remove( block_file )
        self._loadTileData()

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

    @property
    def xlim(self):
        if self._xlim is None:
            tr = self.transform.params.flatten()
            self._xlim = [tr[2], tr[2] + tr[0] * (self._data.shape[2])]
        return self._xlim

    @property
    def ylim(self):
        if self._ylim is None:
            tr = self.transform.params.flatten()
            self._ylim = [tr[5] + tr[4] * (self._data.shape[1]), tr[5]]
        return self._ylim

    def extent(self, epsg: int = None ) -> List[float]:   # left, right, bottom, top
        if epsg is None:    x, y =  self.xlim, self.ylim
        else:               x, y =  self.project_extent( self.xlim, self.ylim, epsg )
        return x + y

    def project_extent(self, xlim, ylim, epsg ):
        inProj = Proj( self.data.attrs['wkt'] )
        outProj = Proj(epsg)
        ylim1, xlim1 = transform(inProj, outProj, xlim, ylim)   # Requires result order reversal- error in transform?
        return xlim1, ylim1

    def inBounds1(self, yc: float, xc: float ) -> bool:
        if (yc < self._ylim[0]) or (yc > self._ylim[1]): return False
        if (xc < self._xlim[0]) or (xc > self._xlim[1]): return False
        return True

    def inBounds(self, yc: float, xc: float ) -> bool:
        if (yc < min(self._ylim)) or (yc > max(self._ylim)): return False
        if (xc < self._xlim[0]) or (xc > self._xlim[1]): return False
        return True

    @property
    def shape(self) -> Tuple[int,int]:
        from spectraclass.data.spatial.tile.manager import TileManager
        return TileManager.instance().block_shape

    @property
    def zeros(self) -> np.ndarray:
        return np.zeros( self.shape, np.int)

    def getBounds(self ) -> Tuple[ Tuple[int,int], Tuple[int,int] ]:
        y0, x0 = self.block_coords[0]*self.shape[0], self.block_coords[1]*self.shape[1]
        return ( y0, y0+self.shape[0] ), ( x0, x0+self.shape[1] )

    def getPointData( self ) -> Tuple[xa.DataArray,Dict]:
        from spectraclass.data.spatial.manager import SpatialDataManager
        if self._point_data is None:
            result: xa.DataArray =  SpatialDataManager.raster2points( self.data )
            self._point_coords: Dict = dict( y=self.data.y, x=self.data.x )
            npts = self.data.y.size * self.data.x.size
            self._point_data = result.assign_coords( samples = np.arange( 0, result.shape[0] ) )
            self._samples_axis = self._point_data.coords['samples']
            self._point_data.attrs['type'] = 'block'
            self._point_data.attrs['dsid'] = self.dsid()
        return (self._point_data, self._point_coords)

    @property
    def point_coords(self) -> Dict[str,xa.DataArray]:
        if self._point_coords is None: self.getPointData()
        return  self._point_coords

    def getSelectedPointData( self, cy: List[float], cx: List[float] ) -> np.ndarray:
        yIndices, xIndices = self.multi_coords2indices(cy, cx)
        return  self.data.values[ :, yIndices, xIndices ].transpose()

    def getSelectedPointIndices( self, cy: List[float], cx: List[float] ) -> np.ndarray:
        yIndices, xIndices = self.multi_coords2indices(cy, cx)
        return  yIndices * self.shape[1] + xIndices

    def getSelectedPoint( self, cy: float, cx: float ) -> np.ndarray:
        index = self.coords2indices(cy, cx)
        return self.data[ :, index['iy'], index['ix'] ].values.reshape(1, -1)

    def plot(self,  **kwargs ) -> xa.DataArray:
        from spectraclass.data.spatial.manager import SpatialDataManager
        color_band = kwargs.pop( 'color_band', None )
        band_range = kwargs.pop( 'band_range', None )
        if color_band is not None:
            plot_data = self.data[color_band]
        elif band_range is not None:
            plot_data = self.data.isel( band=slice( band_range[0], band_range[1] ) ).mean(dim="band", skipna=True)
        else:
            plot_data =  SpatialDataManager.getRGB(self.data)
        SpatialDataManager.plotRaster( plot_data, **kwargs )
        return plot_data

    def coords2indices(self, cy, cx) -> Dict:
        coords = self.transform.inverse(np.array([[cx, cy], ]))
        return dict( iy =math.floor(coords[0, 1]), ix = math.floor(coords[0, 0]) )

    def multi_coords2indices(self, cy: List[float], cx: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.array( list( zip( cx, cy ) ) )
        trans_coords = np.floor(self.transform.inverse(coords))
        indices = trans_coords.transpose().astype( np.int32 )
        return indices[1], indices[0]

    def indices2coords(self, iy, ix) -> Dict:
        (iy,ix) = self.transform(np.array([[ix+0.5, iy+0.5], ]))
        return dict( iy = iy, ix = ix )

    def pindex2coords(self, point_index: int) -> Dict:
        xs = self.point_coords['x'].size
        pi = dict( x= point_index % xs,  y= point_index // xs )
        try:
            return { c: self.point_coords[c].data[ pi[c] ] for c in ['y','x'] }
        except Exception as err:
            lgm().log( f" --> pindex2coords Error: {err}, pid = {point_index}, coords = {pi}" )

    def pindex2indices(self, point_index: int) -> Dict:
        xs = self.point_coords['x'].size
        pi = dict( x= point_index % xs,  y= point_index // xs )
        try:
            selected_sample: List = [ self.point_coords[c].data[ pi[c] ] for c in ['y','x'] ]
            return self.coords2indices( selected_sample[0], selected_sample[1] )
        except Exception as err:
            lgm().log( f" --> pindex2indices Error: {err}, pid = {point_index}, coords = {pi}" )

    def indices2pindex( self, iy, ix ) -> int:
        return self.index_array.values[ iy, ix ]

    def coords2pindex( self, cy, cx ) -> int:
        try:
            index = self.coords2indices( cy, cx )
            return self.index_array.values[ index['iy'], index['ix'] ]
        except IndexError as err:
            return -1

    def multi_coords2pindex(self, ycoords: List[float], xcoords: List[float] ) -> np.ndarray:
        ( yi, xi ) = self.multi_coords2indices( ycoords, xcoords )
        return self.index_array.values[ yi, xi ]


