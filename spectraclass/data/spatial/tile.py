from skimage.transform import ProjectiveTransform
import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
from pyproj import Proj, transform
from spectraclass.data.base import DataManager, DataType
from spectraclass.data.spatial.manager import SpatialDataManager
import os, math, pickle
import traitlets.config as tlc
import traitlets as tl
from spectraclass.model.base import SCConfigurable


def dms() -> SpatialDataManager:  return SpatialDataManager.instance()
# def dm():  return DataManager.instance()

class Tile(tlc.Configurable):

    block_size = tl.Int(250).tag(config=True, sync=True)
    block_shape = tl.List(tl.Int,(250,250),2,2).tag(config=True, sync=True)
    block_dims = tl.List(tl.Int,(4,4),2,2).tag(config=True, sync=True)


    def __init__(self, **kwargs ):
        super(Tile, self).__init__()
        self._data: Optional[xa.DataArray] = None
        self._transform: Optional[ProjectiveTransform] = None
        self.subsampling: int =  kwargs.get('subsample',1)

    @property
    def data(self) -> xa.DataArray:
        if self._data is None:
            self._data: xa.DataArray = dms().getTileData(  **self.config )
        return self._data


    def reset(self):
        self._data = None

    @property
    def name(self) -> str:
        return dms().tileFileName()

    @property
    def transform(self) -> Optional[ProjectiveTransform]:
        if self.data is None: return None
        if self._transform is None:
            self._transform = ProjectiveTransform( np.array(list(self.data.attrs['transform']) + [0, 0, 1]).reshape(3, 3) )
        return self._transform

    def get_block_transform( self, iy, ix ) -> ProjectiveTransform:
        tr0 = self.data.attrs['transform']
        iy0, ix0 = iy * self.block_shape[0], ix * self.block_shape[1]
        y0, x0 = tr0[5] + iy0 * tr0[4], tr0[2] + ix0 * tr0[0]
        tr1 = [ tr0[0], tr0[1], x0, tr0[3], tr0[4], y0, 0, 0, 1  ]
        print( f"Tile transform: {tr0}, Block transform: {tr1}, tile indices = [{self.tile_index}], block indices = [ {iy}, {ix} ]" )
        return  ProjectiveTransform( np.array(tr1).reshape(3, 3) )

    @property
    def filename(self) -> str:
        return self.data.attrs['filename']

    @property
    def nBlocks(self) -> List[ List[int] ]:
        return [ self.data.shape[i+1]//self.block_shape[i] for i in range(2) ]

    def getBlock(self, iy: int, ix: int, **kwargs ) -> Optional["Block"]:
        if self.data is None: return None
        block = Block( self, iy, ix, **kwargs )
        return block

    # def getPointData( self, **kwargs ) -> xa.DataArray:
    #     subsample = kwargs.get( 'subsample', None )
    #     if subsample is None: subsample = self.subsampling
    #     point_data = dms().raster2points( self.data )
    #     return point_data[::subsample]

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
        self.data = self._getData()
        self.transform = tile.get_block_transform( iy, ix )
        self.index_array: xa.DataArray = self.get_index_array()
        self._flow = None
        self._samples_axis: Optional[xa.DataArray] = None
        tr = self.transform.params.flatten()
        self.data.attrs['transform'] = self.transform
        self._xlim = [ tr[2], tr[2] + tr[0] * (self.data.shape[2]) ]
        self._ylim = [ tr[5] + tr[4] * (self.data.shape[1]), tr[5] ]
        self._point_data = None

    @property
    def dsid( self ):
        return "-".join( [ self.tile.name ] + [ str(i) for i in self.block_coords ] )

    def _getData( self ) -> Optional[xa.DataArray]:
        if self.tile.data is None: return None
        ybounds, xbounds = self.getBounds()
        block_raster = self.tile.data[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
        block_raster.attrs['block_coords'] = self.block_coords
        block_raster.name = f"{self.tile.name}_b-{self.block_coords[0]}-{self.block_coords[1]}"
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
        return self.tile.block_shape

    def getBounds(self ) -> Tuple[ Tuple[int,int], Tuple[int,int] ]:
        y0, x0 = self.block_coords[0]*self.shape[0], self.block_coords[1]*self.shape[1]
        return ( y0, y0+self.shape[0] ), ( x0, x0+self.shape[1] )

    def getPointData( self, **kwargs ) -> xa.DataArray:
        dstype = kwargs.get('dstype', DataType.Embedding)
        if dstype == DataType.Embedding:
            if self._point_data is None:
                subsample = kwargs.get( 'subsample', None )
                result: xa.DataArray =  dms().raster2points( self.data )
                if result.size > 0:
                    ptData: xa.DataArray = result if subsample is None else result[::subsample]
                    self._point_data =  self.reduce( ptData )
                else:
                    self._point_data = result
                self._samples_axis = self._point_data.coords['samples']
                self._point_data.attrs['dsid'] = "-".join( [ str(i) for i in self.block_coords ] )
                self._point_data.attrs['type'] = 'block'
            return self._point_data
        elif dstype == DataType.Plot:
            subsample = kwargs.get('subsample', None)
            result: xa.DataArray = dms().raster2points(self.data)
            if result.size > 0:     point_data = result if subsample is None else result[::subsample]
            else:                   point_data = result
            point_data.attrs['dsid'] = "-".join([str(i) for i in self.block_coords])
            point_data.attrs['type'] = 'block'
            return point_data

    def reduce(self, data: xa.DataArray):
        from spectraclass.reduction.embedding import ReductionManager
        if dms().reduce_method != "":
            dave, dmag =  data.values.mean(0), 2.0*data.values.std(0)
            normed_data = ( data.values - dave ) / dmag
            reduced_spectra, reproduction = ReductionManager.instance().reduce( normed_data, dms().reduce_method, dms().model_dims, dms().reduce_nepochs )
            coords = dict( samples=data.coords['samples'], band=np.arange(dms().model_dims) )
            return xa.DataArray( reduced_spectra, dims=['samples', 'band'], coords=coords )
        return data

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
            plot_data =  dms().getRGB(self.data)
        dms().plotRaster( plot_data, **kwargs )
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

class TileManager(tlc.SingletonConfigurable, SCConfigurable):
    tile_size = tl.Int(1000).tag(config=True, sync=True)
    tile_index = tl.List(tl.Int, (0, 0), 2, 2).tag(config=True, sync=True)
    tile_shape = tl.List(tl.Int,(1000,1000),2,2).tag(config=True, sync=True)
    tile_dims = tl.List(tl.Int,(4,4),2,2).tag(config=True, sync=True)

    def __init__(self, **kwargs):
        tlc.SingletonConfigurable.__init__(self)
        SCConfigurable.__init__(self)
        self._tiles: Dict[List, Tile] = {}

    @property
    def iy(self):
        return self.tile_index[0]

    @property
    def ix(self):
        return self.tile_index[1]

    @property
    def tile(self) -> Tile:
        return self._tiles.setdefault(tuple(self.tile_index), Tile())

    def getTileBounds(self ) -> Tuple[ Tuple[int,int], Tuple[int,int] ]:
        y0, x0 = self.iy*self.tile_shape[0], self.ix*self.tile_shape[1]
        return ( y0, y0+self.tile_shape[0] ), ( x0, x0+self.tile_shape[1] )