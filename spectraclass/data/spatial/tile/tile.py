from skimage.transform import ProjectiveTransform
import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
from pyproj import Proj, transform
from spectraclass.data.base import DataManager, DataType
from spectraclass.data.spatial.manager import SpatialDataManager
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
    def filename(self) -> str:
        return self.data.attrs['filename']

    def getBlock(self, iy: int, ix: int, **kwargs ) -> Optional["Block"]:
        if self.data is None: return None
        block = Block( self, iy, ix, **kwargs )
        return block

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
        self.data: Optional[xa.DataArray] = self._getData()
        self.index_array: xa.DataArray = self.get_index_array()
        self._flow = None
        self._samples_axis: Optional[xa.DataArray] = None
        tr = self.transform.params.flatten()
        self.data.attrs['transform'] = self.transform.params.flatten().tolist()
        self._xlim = [ tr[2], tr[2] + tr[0] * (self.data.shape[2]) ]
        self._ylim = [ tr[5] + tr[4] * (self.data.shape[1]), tr[5] ]
        self._point_data: Optional[xa.DataArray] = None
        self._point_coords: Optional[xa.DataArray] = None

    @property
    def transform( self ):
        from spectraclass.data.spatial.tile.manager import TileManager
        return TileManager.instance().get_block_transform( *self.block_coords )

    @property
    def dsid( self ):
        from spectraclass.data.spatial.tile.manager import TileManager
        return "-".join( [ TileManager.instance().tileFileName() ] + [ str(i) for i in self.block_coords ] )

    def _getData( self ) -> Optional[xa.DataArray]:
        from spectraclass.data.spatial.tile.manager import TileManager
        if self.tile.data is None: return None
        tile_name = TileManager.instance().tileFileName(False)
        ybounds, xbounds = self.getBounds()
        block_raster = self.tile.data[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
        block_raster.attrs['block_coords'] = self.block_coords
        block_raster.name = f"{tile_name}_b-{self.block_coords[0]}-{self.block_coords[1]}"
        return block_raster

    def get_index_array(self) -> xa.DataArray:
        stacked_data: xa.DataArray = self.data.stack( samples=self.data.dims[-2:] )
        filtered_samples = stacked_data[1].dropna( dim="samples" )
        indices = np.arange(filtered_samples.shape[0])
        point_indices = xa.DataArray( indices, dims=['samples'], coords=dict(samples=filtered_samples.samples) )
        result = point_indices.reindex( samples=stacked_data.samples, fill_value= -1 )
        return result.unstack()

    @property
    def xlim(self): return self._xlim

    @property
    def ylim(self): return self._ylim

    def extent(self, epsg: int = None ) -> List[float]:   # left, right, bottom, top
        if epsg is None:    x, y =  self.xlim, self.ylim
        else:               x, y =  self.project_extent( self.xlim, self.ylim, 4326 )
        return x + y

    def project_extent(self, xlim, ylim, epsg ):
        inProj = Proj(self.data.spatial_ref.crs_wkt)
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

    def getBounds(self ) -> Tuple[ Tuple[int,int], Tuple[int,int] ]:
        y0, x0 = self.block_coords[0]*self.shape[0], self.block_coords[1]*self.shape[1]
        return ( y0, y0+self.shape[0] ), ( x0, x0+self.shape[1] )

    def getPointData( self, **kwargs ) -> Tuple[xa.DataArray,xa.DataArray]:
        if self._point_data is None:
            subsample = kwargs.get( 'subsample', 1 )
            result: xa.DataArray =  SpatialDataManager.raster2points( self.data )
            self._point_coords: xa.DataArray = result.samples
            self._point_data = result.assign_coords( samples = np.arange( 0, self._point_coords.shape[0] ) )
            if ( self._point_data.size > 0 ):  self._point_data = self._point_data[::subsample]
            self._samples_axis = self._point_data.coords['samples']
            self._point_data.attrs['type'] = 'block'
            self._point_data.attrs['dsid'] = result.attrs['dsid']
        return (self._point_data, self._point_coords)

    @property
    def samples_axis(self) -> xa.DataArray:
        if self._samples_axis is None: self.getPointData()
        return  self._samples_axis

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
        try:
            selected_sample: List = self.samples_axis.values[point_index]
            return dict( y = selected_sample[0], x = selected_sample[1] )
        except Exception as err:
            print( f" --> pindex2coords Error: {err}" )

    def pindex2indices(self, point_index: int) -> Dict:
        try:
            selected_sample: List = self.samples_axis.values[point_index]
            return self.coords2indices( selected_sample[0], selected_sample[1] )
        except Exception as err:
            print( f" --> pindex2coords Error: {err}" )

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


