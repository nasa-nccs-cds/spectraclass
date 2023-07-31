import traceback
from skimage.transform import ProjectiveTransform
import numpy as np
from osgeo import ogr, osr
from os import path
import cartopy.crs as crs
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from pyproj import Transformer
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
import os, math, pickle, time

def combine_masks( mask1: Optional[np.ndarray], mask2: Optional[np.ndarray] ) -> Optional[np.ndarray]:
    if mask1 is None: return mask2
    if mask2 is None: return mask1
    return mask1 & mask2

def stat( data: xa.DataArray ) -> str:
    x: np.ndarray = data.values
    return f"({np.nanmean(x):.2f}, {np.nanstd(x):.2f}, {np.count_nonzero( np.isnan(x) ):.2f})"

def nnan( data: xa.DataArray ) -> int:
    return np.count_nonzero( np.isnan(data.values) )

def size( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return "NONE" if (array is None) else array.size

def shp( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return "NONE" if (array is None) else array.shape

def nz( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return "NONE" if array is None else np.count_nonzero(array)

def nvalid( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return "NONE" if array is None else np.count_nonzero( ~np.isnan(array) )

def nnan( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return "NONE" if array is None else np.count_nonzero( np.isnan(array) )

def xarange( data: xa.DataArray, axis=None ) -> Tuple[np.ndarray,np.ndarray]:
    return ( np.nanmin(data.values,axis=axis), np.nanmax(data.values,axis=axis) )

def parse_p4( p4str: str ) -> Dict[str,str]:
    p4d = {}
    for p4entry in p4str.split('+'):
        entry = p4entry.strip()
        if '=' in entry:
            toks = entry.split('=')
            p4d[toks[0]] = toks[1]
        elif len(entry):
            p4d[entry] = 'true'
    return p4d

class DataContainer:

    def __init__(self, **kwargs):
        super(DataContainer, self).__init__()
        self._data_projected = kwargs.get( 'data_projected', False )
        self.initialize()

    def initialize(self):
        self._extent: List[float] = None
        self._transform: List[float] = None
        self._ptransform: ProjectiveTransform = None
        self._data: Optional[xa.DataArray] = None
        self._transformer: Transformer = None
        self._xlim = None
        self._ylim = None

    @property
    def transformer(self) -> Transformer:
        from spectraclass.data.spatial.tile.manager import TileManager
        if self._transformer is None:
            self._transformer = Transformer.from_crs( self.data.spatial_ref.crs_wkt, f'epsg:{TileManager.ESPG}' )
        return self._transformer

    @property
    def data(self) -> Optional[xa.DataArray]:
        if self._data is None:
            self._data = self._get_data()
        sz = size( self._data )
        return None if (sz==0) else self._data

    def _get_data(self) -> xa.DataArray:
        raise NotImplementedError(f" Attempt to call abstract method _get_data on {self.__class__.__name__}")

    @property
    def grid_size(self) -> int:
        return self.data.shape[-1]*self.data.shape[-2]

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

    def wkt_to_proj4(self, wkt_text: str, as_dict=True ) -> Union[str,Dict[str,str]]:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt_text)
        p4str = srs.ExportToProj4()
        return parse_p4( p4str ) if as_dict else p4str

    @property
    def projection(self) -> crs.Projection:
        return crs.Projection(self.proj4)

    @property
    def wkt(self) -> str:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return tm().tile.spatial_ref.attrs['crs_wkt']

    @property
    def proj4(self) -> str:
        return self.wkt_to_proj4(self.wkt)

    def get_extent(self, projection: crs.Projection = None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        ext = self.extent
        if projection is None:
            return  ( (ext[0],ext[2]), (ext[1],ext[3]) )
        else:
            trans_ext = projection.transform_points( self.projection, np.array(ext[:2]), np.array(ext[2:]) )
            return ( tuple(trans_ext[:,0].tolist()), tuple(trans_ext[:,1].tolist()) )

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
    def xlim(self) -> Tuple[float,float]:
        if self._xlim is None:
            xc: np.ndarray = self.xcoord
            dx = (xc[-1]-xc[0])/(xc.size-1)
            self._xlim = ( xc[0]-dx, xc[-1]+dx )
        return self._xlim

    @property
    def bounds(self) -> Tuple[float,float,float,float]:
        return self.xlim + self.ylim

    @property
    def ylim(self) -> Tuple[float,float]:
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

    def __init__(self, tile_index: int, **kwargs ):
        super(Tile, self).__init__(**kwargs)
        self._blocks = {}
        self.spatial_ref: xa.DataArray = None
        self._index = tile_index
        self.subsampling: int =  kwargs.get('subsample',1)
        self._mean = None
        self._std = None

    @property
    def index(self):
        return self._index

    def _get_data(self) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        tile_data = tm().getTileData()
        lgm().log( f"#Tile[{self._index}]-> Read Data: shape = {tile_data.shape}, dims={tile_data.dims}", print=True )
        tile_data = self.filter_degraded_bands( tile_data )
        return tile_data

    def filter_degraded_bands(self, tile_data: xa.DataArray ) -> xa.DataArray:
        lgm().log(f"  ---> attrs: {tile_data.attrs}" )
        init_shape = tile_data.shape
        if '_FillValue' in tile_data.attrs:
            nodata = tile_data.attrs['_FillValue']
            tile_data = tile_data if np.isnan(nodata) else tile_data.where(tile_data != nodata, np.nan)
        point_data = tile_data.stack(samples=tile_data.dims[-2:]).transpose()
        bvcnts = [nnan(point_data.values[:, ic]) for ic in range(point_data.shape[1])]
        bmask: np.ndarray = (np.array(bvcnts) < point_data.shape[0] * 0.01)
        tile_data = tile_data[bmask]
        lgm().log( f"-------------\n         ***** Filtering invalid tile bands, init_shape = {init_shape}, resulting Tile shape = {tile_data.shape}")
        return tile_data

    @property
    def name(self) -> str:
        return self.data.attrs['tilename']

    def getDataBlock(self, ix: int, iy: int, **kwargs ) -> Optional["Block"]:
        if (ix,iy) in self._blocks: return self._blocks[ (ix,iy) ]
        return self._blocks.setdefault( (ix,iy), Block( self, ix, iy, self._index, **kwargs ) )

    @log_timing
    def getBlocks(self, **kwargs ) -> List["Block"]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.base import dm
        block_selection: Optional[Dict] = kwargs.get( 'selection', dm().modal.get_block_selection() )
        data_blocks = []
        for ix in range(0, tm().block_dims[0]):
            for iy in range(0, tm().block_dims[1]):
                if (block_selection is None) or ((ix,iy) in block_selection):
                    data_blocks.append( self.getDataBlock( ix, iy, **kwargs )  )
        lgm().log( f"getBlocks: tile_shape[x,y]={tm().tile_shape},  block_dims[x,y]={tm().block_dims}, "
                   f"raster_shape={tm().tile.data.shape}, nblocks active = {len(data_blocks)}")
        return data_blocks

    @log_timing
    def band_data(self, iband: int, **kwargs ) -> xa.DataArray:
        rv = self.data[ iband, :, : ]
        return rv

    def rgb_data(self, bands: Tuple[int,int,int], **kwargs ) -> xa.DataArray:
        norm = kwargs.get('norm',False)
        dim = self.data.dims
        slices: List[xa.DataArray] = [ self.data[ iband, :, : ] for iband in bands ]
        rgb: xa.DataArray = xa.concat( slices, dim=dim[0] )
        lgm().log( f"RGB: rgb.shape={rgb.shape}, dims={rgb.dims}")
        if norm:
            dmin, dmax = np.nanmin(rgb.values), np.nanmax(rgb.values)
            lgm().log(f"RGB: rgb dmin={dmin}, dmax={dmax}")
            rgb = (rgb-dmin)/(dmax-dmin)
        return rgb.transpose( dim[1], dim[2], dim[0] )

    @log_timing
    def block_slice_data(self, iband: int, xbounds: Tuple[int,int], ybounds: Tuple[int,int] ) -> np.ndarray:
        return self.data[ iband, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ].to_numpy().squeeze()

    # @log_timing
    # def saveMetadata_preread( self ):
    #     from spectraclass.data.base import DataManager, dm
    #     file_path = dm().metadata_file
    #     block_data: Dict[Tuple,int] = {}
    #     blocks: List["Block"] = self.getBlocks()
    #     lgm().log(f"------------ saveMetadata: raster shape = {self.data.shape}" )
    #     raster_band: np.ndarray = self.band_data( 0 )
    #     nodata = self.data.attrs.get('_FillValue')
    #     for block in blocks:
    #         xbounds, ybounds = block.getBounds()
    #         raster_slice: np.ndarray = raster_band[ ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
    #         lgm().log( f"   * READ raster slice[{block.block_coords}], xbounds={xbounds}, ybounds={ybounds}, slice shape={raster_slice.shape}, nodata val = {nodata}")
    #         if not np.isnan(nodata):
    #             raster_slice[ raster_slice == nodata ] = np.nan
    #         nodata_mask = np.isnan( raster_slice )
    #         block_data[ block.block_coords ] = np.count_nonzero(nodata_mask)
    #
    #     os.makedirs( os.path.dirname(file_path), exist_ok=True )
    #     try:
    #         with open( file_path, "w" ) as mdfile:
    #             for (k,v) in self.data.attrs.items():
    #                 mdfile.write( f"{k}={v}\n" )
    #             for bcoords, bsize in block_data.items():
    #                 mdfile.write( f"{self.bsizekey(bcoords)}={bsize}\n" )
    #         lgm().log(f" ---> Writing metadata file: {file_path}", print=True)
    #     except Exception as err:
    #         lgm().log(f" ---> ERROR Writing metadata file at {file_path}: {err}", print=True)
    #         if os.path.isfile(file_path): os.remove(file_path)


    @log_timing
    def saveMetadata( self ):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        t0 = time.time()
        file_path = dm().metadata_file
        if tm().reprocess or not os.path.isfile(file_path):
            block_data: Dict[Tuple,int] = {}
            blocks: List["Block"] = self.getBlocks()
            print( f"Generating metadata: {dm().modal.image_name}.mdata.txt" )
            nodata = tm().tile.data.attrs.get('_FillValue')
            for block in blocks:
                xbounds, ybounds = block.getBounds()
                try:
                    raster_slice: np.ndarray = tm().tile.data[ 0, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ].to_numpy().squeeze().astype(np.float32)
                    if not np.isnan(nodata):
                        lgm().log(f"----> processing raster block, shape = {raster_slice.shape}, dtype={raster_slice.dtype}, nodata={nodata}")
                        raster_slice[ raster_slice == nodata ] = np.nan
                    valid_mask = ~np.isnan( raster_slice )
                    block_data[ block.block_coords ] = np.count_nonzero(valid_mask)
                except Exception as err:
                    lgm().exception( f"Error processing block{block.block_coords}: xbounds={xbounds}, ybounds={ybounds}, base shape = {tm().tile.data.shape}")
            os.makedirs( os.path.dirname(file_path), exist_ok=True )
            try:
                with open( file_path, "w" ) as mdfile:
                    for (k,v) in self.data.attrs.items():
                        mdfile.write( f"{k}={v}\n" )
                    for bcoords, bsize in block_data.items():
                        mdfile.write( f"{self.bsizekey(bcoords)}={bsize}\n" )
                lgm().log(f" ---> Writing metadata, time = {(time.time()-t0)/60} min", print=True)
            except Exception as err:
                lgm().log(f" ---> ERROR Writing metadata file at {file_path}: {err}", print=True)
                if os.path.isfile(file_path): os.remove(file_path)
        else:
            print(f"Skipping existing: {dm().modal.image_name}.mdata.txt" )

#     @log_timing
#     def saveMetadata_parallel( self ):
#         from multiprocessing import cpu_count, get_context, Pool
#         from spectraclass.data.base import DataManager, dm
#         file_path = dm().metadata_file
#         block_data: Dict[Tuple,int] = {}
#         blocks: List["Block"] = self.getBlocks()
#         lgm().log(f"------------ saveMetadata: raster shape = {self.data.shape}" )
#         nodata = self.data.attrs.get('_FillValue')
#
# #        with get_context("spawn").Pool(processes=nproc) as p:
# #            image_tiles = p.map(image_processor, image_ids)
#
#         for block in blocks:
#             xbounds, ybounds = block.getBounds()
#             raster_slice: np.ndarray = self.block_slice_data( 0, xbounds, ybounds )
#             lgm().log( f"   * READ raster slice[{block.block_coords}], xbounds={xbounds}, ybounds={ybounds}, slice shape={raster_slice.shape}")
#             if not np.isnan(nodata):
#                 raster_slice[ raster_slice == nodata ] = np.nan
#             nodata_mask = np.isnan( raster_slice )
#             block_data[ block.block_coords ] = np.count_nonzero(nodata_mask)
#
#         os.makedirs( os.path.dirname(file_path), exist_ok=True )
#         try:
#             with open( file_path, "w" ) as mdfile:
#                 for (k,v) in self.data.attrs.items():
#                     mdfile.write( f"{k}={v}\n" )
#                 for bcoords, bsize in block_data.items():
#                     mdfile.write( f"{self.bsizekey(bcoords)}={bsize}\n" )
#             lgm().log(f" ---> Writing metadata file: {file_path}", print=True)
#         except Exception as err:
#             lgm().log(f" ---> ERROR Writing metadata file at {file_path}: {err}", print=True)
#             if os.path.isfile(file_path): os.remove(file_path)

    def bsizekey(self, bcoords: Tuple[int,int] ) -> str:
        return f"nvalid-{bcoords[0]}-{bcoords[1]}"

    def bskey2coords(self, bskey: str ) -> Tuple[int,int]:
        toks = bskey.split("-")
        return ( int(toks[1]), int(toks[2]) )

class ThresholdRecord:

    def __init__(self, fdata: xa.DataArray, block_coords: Tuple[int,int], image_index: int ):
        lgm().log( f"#TR: Create ThresholdRecord[{image_index}:{block_coords}] ")
        self._tmask: xa.DataArray = None
        self._block_coords: Tuple[int,int] = block_coords
        self._image_index: int = image_index
        self.thresholds: Tuple[float,float] = (0.0,1.0)
        self.fixed: bool = False
        self.needs_update = [ True, True ]
        self.fdata: xa.DataArray = fdata
        self._drange = None

    def applicable(self, block_coords: Tuple[int,int] ) -> bool:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return ( self._image_index == tm().image_index ) and ( self._block_coords == block_coords )

    @property
    def tmask(self) -> Optional[xa.DataArray]:
        self.compute_mask()
        return self._tmask

    @property
    def drange(self):
        if self._drange is None:
            self._drange = [ np.nanmin(self.fdata.values), np.nanmax(self.fdata.values) ]
            self._drange.append( self.drange[1]-self.drange[0] )
        return self._drange

    def set_thresholds( self, thresholds: Tuple[float,float] ) -> np.ndarray:
        self.needs_update = [ self.thresholds[i] != thresholds[i] for i in (0,1) ]
        self.thresholds = thresholds
        return self.compute_mask()

    def clear(self):
        self._tmask = None
        self.thresholds = (0.0, 1.0)

    def is_empty(self):
        return (self._tmask is None)

    def compute_mask(self) -> np.ndarray:
        mask = None
        if self.needs_update[1]:
            thresh = self.drange[0] + self.drange[2]*self.thresholds[1]
            mask = ( self.fdata.values > thresh )
        if self.needs_update[0]:
            thresh = self.drange[0] + self.drange[2]*self.thresholds[0]
            lmask = ( self.fdata.values < thresh )
            mask = (lmask & mask) if (mask is not None) else lmask
        if mask is not None:
            self._tmask = self.fdata.copy( data=mask )
            self.needs_update = [ False, False ]
        return mask

class Block(DataContainer):

    def __init__(self, tile: Tile, ix: int, iy: int, itile: int, **kwargs ):
        super(Block, self).__init__( data_projected=True, **kwargs )
        self.initialize()
        self.init_task = None
        self._index_array: xa.DataArray = None
        self._gid_array: np.ndarray = None
        self._flow = None
        self._samples_axis: Optional[xa.DataArray] = None
        self._point_data: Optional[xa.DataArray] = None
        self._filtered_point_data: Optional[xa.DataArray] = None
        self._point_coords: Optional[Dict[str,np.ndarray]] = None
        self._point_mask: Optional[np.ndarray] = None
        self._raster_mask: Optional[np.ndarray] = None
        self._tmask: np.ndarray = None
        self._model_data: xa.DataArray = None
        self._reproduction: xa.DataArray = None
        self._reduction_input_data: xa.DataArray = None
        self.tile: Tile = tile
        self.config = kwargs
        self._trecs: Tuple[ Dict[int,ThresholdRecord], Dict[int,ThresholdRecord] ] = ( {}, {} )
        self.block_coords: Tuple[int,int] = (ix,iy)
        self.tile_index = itile

    def set_thresholds(self, bUseModel: bool, iFrame: int, thresholds: Tuple[float,float] ) -> bool:
        trec: ThresholdRecord = self.threshold_record( bUseModel, iFrame )
        initialized = trec.is_empty()
        mask = trec.set_thresholds( thresholds )
        lgm().log(f"#IA: MASK[{iFrame}].set_thresholds---> nmasked pixels = {np.count_nonzero(mask)} ")
        self._tmask = None
        self._index_array = None
        self.tile.initialize()
        return initialized

    @property
    def index(self) -> Tuple[int,Tuple]:
        return (self.tile_index, self.block_coords)

    @property
    def cindex(self) -> Tuple[int,int,int]:
        return ( self.tile_index, self.block_coords[0], self.block_coords[1]  )

    def threshold_record(self, model_data: bool, iFrame: int ) -> ThresholdRecord:
        from spectraclass.data.base import dm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        trecs: Dict[int,ThresholdRecord] = self._trecs[ int(model_data) ]
        if iFrame in trecs: return trecs[ iFrame ]
        fdata: xa.DataArray = self.points2raster( dm().getModelData() ) if model_data else self.data
        return trecs.setdefault( iFrame,  ThresholdRecord( fdata[iFrame], self.block_coords, tm().image_index ) )

    def get_trec(self, model_data: bool, iFrame: int ) -> Optional[ThresholdRecord]:
        trecs: Dict[int,ThresholdRecord] = self._trecs[ int(model_data) ]
        return trecs.get( iFrame, None )


    def get_mask_list(self, current_frame = -1 ) -> Tuple[ List[str], str ]:
        mask_list, types, value = [], ["band", "model" ], None
        for ttype, trecs in zip(types,self._trecs):
            for iFrame, trec in trecs.items():
                if trec.applicable(self.block_coords) and (trec.tmask is not None):
                    mask_name = f"{ttype}:{iFrame}"
                    mask_list.append( mask_name )
                    if iFrame == current_frame:
                        value = mask_name
        return ( mask_list, value )

    @exception_handled
    def get_threshold_mask( self, raster=False, reduced=True ) -> np.ndarray:
        if self._tmask is None:
            ntmask = None
            for trecs in self._trecs:
                for iFrame, trec in trecs.items():
                    if trec.tmask is not None:
                        ntmask = trec.tmask.values if (ntmask is None) else (ntmask | trec.tmask.values)
            if ntmask is not None:
                self._tmask = ~ntmask
                self._tmask = self._tmask & self.raster_mask
        if self._tmask is not None:
            if not raster:
                ptmask = self._tmask.flatten()
                if reduced: ptmask = ptmask[self._point_mask]
                return ptmask
        return self._tmask

    @exception_handled
    def get_threshold_mask1( self, raster=False, reduced=True ) -> Optional[np.ndarray]:
        if self._tmask is None:
            ntmask = None
            for trecs in self._trecs:
                for iFrame, trec in trecs.items():
                    if trec.tmask is not None:
                        ntmask = trec.tmask.values if (ntmask is None) else (ntmask | trec.tmask.values)
            if ntmask is not None:
                self._tmask = ~ntmask
            #     self._tmask = self._tmask & self.raster_mask
            # else:
            #     self._tmask = self.raster_mask
        if (not raster) and (self._tmask is not None):
            ptmask = self._tmask.flatten()
            if reduced: ptmask = ptmask[self._point_mask]
            return ptmask
        return self._tmask

    def classmap(self, default_value: int =0 ) -> xa.DataArray:
        return xa.full_like( self.data[0].squeeze(drop=True), default_value, dtype=np.int32 )

    @property
    def gid_array(self):
        if self._gid_array is None:
            self._gid_array = self.get_gid_array()
        return self._gid_array

    def dsid( self ):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return "-".join( [ tm().tileName() ] + [ str(i) for i in self.block_coords ] )

    def validate_parameters(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        assert ( self.block_coords[0] < tm().block_dims[0] ) and ( self.block_coords[1] < tm().block_dims[1] ), f"Block coordinates {self.block_coords} out of bounds with block dims = {tm().block_dims}"

    def translate_transform(self, transform: List[float] ) -> List[float]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        dx, dy = [ self.block_coords[0]*tm().block_size, self.block_coords[1]*tm().block_size ]
        x0 = transform[2] + dx * transform[0] + dy * transform[1]
        y0 = transform[5] + dx * transform[3] + dy * transform[4]
        return [ transform[0], transform[1], x0, transform[3], transform[4], y0 ]

    def get_raw_data(self) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        (xbounds, ybounds), tile_data = self.getBounds(), self.tile.data
        raster_slice = tile_data[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        return raster_slice if (raster_slice.size == 0) else TileManager.filter_invalid_data( raster_slice )

    @log_timing
    def _get_data( self ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        raw_raster: Optional[xa.DataArray] = self.load_block_raster()
        if raw_raster is None:
            if self.tile.data is None: return None
            xbounds, ybounds = self.getBounds()
            raster_slice = self.tile.data[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
            raster_slice.attrs['transform'] = self.translate_transform( raster_slice.attrs['transform'] )
            raw_raster = raster_slice if (raster_slice.size == 0) else TileManager.process_tile_data( raster_slice )
            lgm().log(f" *** BLOCK{self.block_coords}: load-slice ybounds={ybounds}, xbounds={xbounds}, raster shape={raw_raster.shape}")
            raw_raster.attrs['anomaly'] = False
        block_raster = raw_raster # self._apply_mask( raw_raster )
        block_raster.attrs['block_coords'] = self.block_coords
        block_raster.attrs['tile_shape'] = tm().tile.data.shape
        block_raster.attrs['block_dims'] = tm().block_dims
        block_raster.attrs['tile_size']  = tm().tile_size
        block_raster.attrs['dsid'] = self.dsid()
        block_raster.attrs['file_name'] = self.file_name
        block_raster.name = self.file_name
        lgm().log( f"IA: block data, shape: {block_raster.shape}, dims: {block_raster.dims}")
        return block_raster

    @property
    def data_file(self):
        from spectraclass.data.base import DataManager, dm
        return dm().modal.dataFile( block=self, index=self.tile_index )

    @log_timing
    def has_data_samples(self) -> bool:
        file_exists = path.isfile(self.data_file)

        # if file_exists:
        #     with xa.open_dataset(self.data_file) as dataset:
        #         if (len(dataset.coords) == 0):
        #             nsamples = 0
        #         elif 'samples' in dataset.coords:
        #             nsamples = dataset.coords['samples'].size
        #         else:
        #             lgm().log(f" BLOCK{self.block_coords}: NO SAMPLES COORD-> coords= {list(dataset.coords.keys())}")
        #             nsamples = 9999
        #         lgm().log( f" BLOCK{self.block_coords} data_samples={nsamples}")
        #         file_exists = (nsamples > 0)
        return file_exists

    def has_data_file(self, non_empty=False ) -> bool:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        file_exists = path.isfile(self.data_file) and tm().load_block_cache
        lgm().log(f" BLOCK{self.block_coords}: file_exists={file_exists}, data file= {self.data_file}")
        # if non_empty and file_exists:
        #     return self.has_data_samples()
        return file_exists

    @log_timing
    def load_block_raster(self) -> Optional[xa.DataArray]:
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        raw_raster: Optional[xa.DataArray] = None
        if self.has_data_file():
            dataset: xa.Dataset = dm().modal.loadDataFile( block=self )
            raw_raster = self.extract_input_data( dataset, raster=True )
            for aid, aval in dataset.attrs.items():
                if aid not in raw_raster.attrs:
                    raw_raster.attrs[aid] = aval
            x,y = raw_raster.x.values, raw_raster.y.values
            lgm().log(f"#LB: @BLOCK{self.block_coords}->get_data: load-datafile raster shape={raw_raster.shape}, exent= ({x[0]},{x[-1]}) ({y[0]},{y[-1]})")
            lgm().log(f"#LB: @BLOCK{self.block_coords}---> raw data attrs = {raw_raster.attrs.keys()}")
            lgm().log(f"#LB: @BLOCK{self.block_coords}---> dset attrs = {dataset.attrs.keys()}")
            if raw_raster.size == 0: ufm().show( "This block does not appear to have any data.", "warning" )
        return raw_raster

    @exception_handled
    def extract_input_data(self, dataset: xa.Dataset, **kwargs ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.learn.pytorch.trainer import stat
        raster = kwargs.get('raster',False)
        raw_data: xa.DataArray = tm().mask_nodata( dataset["raw"] )
        point_data = self.raster2points( raw_data, norm=True, **kwargs)
        baseline_spectrum: xa.DataArray = dataset.get('baseline', None )
        result = point_data
        if baseline_spectrum is not None:
            sdiff: xa.DataArray = point_data - baseline_spectrum
            result = tm().norm( sdiff )
            lgm().log( f"#ANOM.TILE.extract_input_data{kwargs}-> input: shape={point_data.shape}, stat={stat(point_data)}; "
                       f"result: shape={result.shape}, raw stat={stat(sdiff)}, norm stat={stat(result)}")
        result.attrs['anomaly'] = (baseline_spectrum is not None)
        if raster: result = self.points2raster(  result, coords=raw_data.coords )
        result.attrs.update( dataset["raw"].attrs )
        return result

    @exception_handled
    def getModelData(self,  **kwargs ) -> xa.DataArray:
        raster = kwargs.pop('raster', True)
        return self.points2raster( self.model_data ) if raster else self.model_data

    @exception_handled
    def getBandData( self, **kwargs ) -> xa.DataArray:
        raster = kwargs.get( 'raster', True)
        self.createPointData(**kwargs)
        if raster:  return self.get_raster_data( **kwargs )
        else:       return self.get_point_data( **kwargs )

    def get_constant_array(self, value: float, **kwargs  ) -> xa.DataArray:
        raster = kwargs.get('raster', True )
        bdata: xa.DataArray = self.data if raster else self.point_data
        return bdata.copy( data=np.full( bdata.shape, value ) )

    def get_point_data( self, **kwargs) -> xa.DataArray:
        class_filter =  kwargs.get( 'class_filter', False)
        return self.filtered_point_data if class_filter else self.point_data

    @property
    def model_data(self) -> xa.DataArray:
        if self._model_data is None: self._get_model_data()
        return self._model_data

    def getReproduction(self, raster=False ) -> xa.DataArray:
        if self._model_data is None: self._get_model_data()
        return self.points2raster(self._reproduction) if raster else self._reproduction

    @property
    def reduction_input_data(self) -> xa.DataArray:
        if self._model_data is None: self._get_model_data()
        return self._reduction_input_data

    def _get_model_data(self):
        from spectraclass.reduction.trainer import mt
        pdata = self.filtered_point_data
        (self._model_data, self._reproduction) = mt().reduce( pdata )
        self._reduction_input_data = pdata
        self._model_data.attrs['block_coords'] = self.block_coords
        self._model_data.attrs['dsid'] = self.dsid()
        self._model_data.attrs['file_name'] = self.file_name
        self._model_data.attrs['pmask'] = pdata.attrs.get('pmask',None)
        self._model_data.name = self.file_name

    @exception_handled
    def _apply_mask( self, block_array: xa.DataArray, raster=False, reduced=False ) -> xa.DataArray:
        lgm().log(f"#IA: apply_mask ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
        mask_array: Optional[np.ndarray] = self.get_threshold_mask( raster=raster, reduced=reduced )
        result = block_array
        if mask_array is not None:
            if raster and block_array.ndim == 3: mask_array = np.expand_dims( mask_array, 0 )
            lgm().log( f"#IA: ~~~~~~~~~~>> shapes-> mask={shp(mask_array)}, data={shp(block_array)}; nz={nz(mask_array)}")
            result = block_array.copy( data=np.where( mask_array, block_array.values, np.nan ) )
            lgm().log( f"#IA: ~~~~~~~~~~>> resulting masked-data shape={shp(result)}; nnan/band={nnan(result.values)//result.shape[0]}")
            result.attrs['threshold_mask'] = mask_array
        return result

    def addTextureBands( self ):
        from spectraclass.features.texture.manager import TextureManager, texm
        self._data = texm().addTextureBands( self.data )

    @property
    def file_name(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return f"{tm().tileName()}_b-{tm().block_size}-{self.block_coords[0]}-{self.block_coords[1]}"

    def get_gid_array(self) -> np.ndarray:
        d0: np.ndarray = self.data.values[0].squeeze().flatten()
        gids = np.arange(d0.size)
        return gids[ ~np.isnan(d0) ]

    def pid2gid(self, pid: int ) -> int:
        return self.gid_array[pid]

    def pids2gids(self, pids: np.ndarray ) -> np.ndarray:
        return self.gid_array[ pids ]

    @property
    def shape(self) -> Tuple[int,...]:
        return self.data.shape

    @property
    def zeros(self) -> np.ndarray:
        return np.zeros( self.shape, np.int32)

    def getBounds(self ) -> Tuple[ Tuple[int,int], Tuple[int,int] ]:
        from spectraclass.data.spatial.tile.manager import tm
        bsize = tm().block_size
        x0, y0 = self.block_coords[0]*bsize, self.block_coords[1]*bsize
        bounds = ( x0, x0+bsize ), ( y0, y0+bsize )
        lgm().log( f"GET BLOCK{self.block_coords} BOUNDS: dx={bounds[0]}, dy={bounds[1]}, block_size={bsize}")
        return bounds

    @property
    def raw_point_data(self) -> Optional[xa.DataArray]:
        if self._point_data is None: self.createPointData()
        return self._point_data

    def get_raster_data(self, **kwargs ) -> Optional[xa.DataArray]:
        class_filter = kwargs.get( 'class_filter', False )
        if self._point_data is None: self.createPointData()
        xptdata = self._point_data
        if class_filter:
            ptdata = np.where( self._point_mask, self._point_data.values, np.nan )
            xptdata = self._point_data.copy( data=ptdata )
        return self.points2raster( xptdata )

    @property
    def point_data(self) -> Optional[xa.DataArray]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if self._point_data is None: self.createPointData()
        return tm().norm( self._point_data )

    @property
    def filtered_point_data(self) -> Optional[xa.DataArray]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if self._point_data is None: self.createPointData()
        return tm().norm( self._point_data[self._point_mask] )

    @property
    def filtered_raster_data(self) -> Optional[xa.DataArray]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if self._point_data is None: self.createPointData()
        fpoint_data: xa.DataArray = tm().norm( self._point_data[self._point_mask] )
        return self.points2raster( fpoint_data )

    @property
    def class_mask(self) -> np.ndarray:
        if self._point_mask is None:
            self.createPointData()
        return self._point_mask

    def point_coords(self)-> Dict[str, np.ndarray]:
        if self._point_data is None: self.createPointData()
        return self._point_coords

    @exception_handled
    def createPointData(self, **kwargs):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.learn.pytorch.trainer import mpt
        self._point_data =  self.raster2points( self.data, **kwargs )
        if self._point_data is not None:
            self._samples_axis = self._point_data.coords['samples']
            normed_data = tm().norm( self._point_data )
            self._point_mask  = mpt().get_class_mask( normed_data )
            lgm().log(f"#FPDM-getPointData: filtered data shape={self._point_data.shape}, "
                      f"cfmask shape={self._point_mask.shape}, nz={np.count_nonzero(self._point_mask)}")
            self._point_data.attrs['type'] = 'block'
            self._point_data.attrs['dsid'] = self.dsid()
            self._point_data.attrs['pmask'] = self._point_mask
            self._point_coords: Dict[str, np.ndarray] = dict(y=self.data.y.values, x=self.data.x.values)

    @property
    def raster_mask(self) -> Optional[np.ndarray]:
        if self._point_data is None: self.createPointData()
        return self._raster_mask

    @property
    def mask(self) -> np.ndarray:
        return self.point_coords['mask']

    def getSelectedPointData( self, cy: List[float], cx: List[float] ) -> np.ndarray:
        yIndices, xIndices = self.multi_coords2indices(cy, cx)
        return  self.data.values[ :, yIndices, xIndices ].transpose()

    def gid2pid(self, gid: int ) -> int:
        pids: np.ndarray = np.asarray(self.gid_array == gid).nonzero()[0]
        return -1 if pids.size == 0 else pids[0]

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

    def gid2coords(self, pid: int) -> Dict:
        pi =self.gid2indices(pid)
        try:
            return { c: self.point_coords[c][ pi[f"i{c}"] ] for c in ['y','x'] }
        except Exception as err:
            lgm().log( f" --> pindex2coords Error: {err}, pid = {pid}, coords = {pi}" )

    def gid2indices(self, pid: int) -> Dict:
        return dict( ix=pid % self.shape[1], iy=pid // self.shape[1] )

        # point_index = self.pid_array[pid]
        # xs = self.point_coords['x'].size
        # pi = dict( x= point_index % xs,  y= point_index // xs )
        # try:
        #     selected_sample: List = [ self.point_coords[c][ pi[c] ] for c in ['y','x'] ]
        #     return self.coords2indices( selected_sample[0], selected_sample[1] )
        # except Exception as err:
        #     lgm().log( f" --> pindex2indices Error: {err}, pid = {point_index}, coords = {pi}" )

    def points2raster(self, points_data: xa.DataArray, **kwargs ) -> xa.DataArray:
        t0 = time.time()
        coords = kwargs.get( 'coords', None )
        if coords is None: coords = self.data.coords
        [x, y] = [ coords[cn].values for cn in ['x','y']]
        dims = [ points_data.dims[1], 'y', 'x' ]
        coords = [(dims[0], points_data[dims[0]].data), ('y', y), ('x',x)]
        rpdata = np.full([x.size * y.size, points_data.shape[1]], float('nan'))
        lgm().log(f"#P2R points2raster:  points_data.attrs = {list(points_data.attrs.keys())}")
        self._point_mask  = points_data.attrs.get('pmask',None)
        if self._point_mask is not None:
            pnz = np.count_nonzero(self.class_mask)
            lgm().log(f"#P2R --> cmask shape = {self.class_mask.shape}")
        else: pnz = -1
        lgm().log(f"#P2R --> points_data, shape={points_data.shape}; pmask #nz={pnz}; raster shape = {rpdata.shape}; ")
        if pnz == points_data.shape[0]:
            rpdata[ self.class_mask ] = points_data.data
        elif rpdata.shape[0] == points_data.shape[0]:
            rpdata = points_data.values.copy()
        else:
            raise Exception( f"Size mismatch: pnz={pnz}, points_data.shape={points_data.shape}, rpdata.shape={rpdata.shape}")
        raster_data = rpdata.transpose().reshape([points_data.shape[1], y.size, x.size])
        lgm().log( f"#P2R points->raster[{self.dsid()}], time= {time.time()-t0:.2f} sec, raster: dims={dims}, "
                   f"shape={raster_data.shape}, nnan = {np.count_nonzero(np.isnan(raster_data))}" )
        rname = kwargs.get( 'name', points_data.name )
        return xa.DataArray( raster_data, coords, dims, rname, points_data.attrs )

    def raster2points(self, base_raster: xa.DataArray, **kwargs) -> Optional[xa.DataArray]:  # base_raster dims: [ band, y, x ]
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        norm = kwargs.get('norm',False)
        if (base_raster is None) or (base_raster.shape[0] == 0): return None
        stacked_data: xa.DataArray = base_raster.stack(samples=base_raster.dims[-2:]).transpose()
        point_data: xa.DataArray = stacked_data.assign_coords({"samples": np.arange(stacked_data.shape[0])})

        if '_FillValue' in point_data.attrs:
            nodata = point_data.attrs.get('_FillValue',np.nan)
            point_data = point_data if np.isnan(nodata) else point_data.where(point_data != nodata, np.nan)

        lgm().log(f"#FPD[{self.block_coords}]: band-filtered point_data shape = {point_data.shape}, nnan={nnan(point_data)} ")
        if point_data.size > 0:
            smean: np.ndarray = np.nanmean( point_data.values, axis=0 )
            for iB in range( smean.size ):
                bmask: np.ndarray = np.isnan( point_data.values[:,iB] )
                point_data[ bmask, iB ] = smean[iB]

        point_data.attrs['dsid'] = base_raster.name
        lgm().log(f"#FPD[{self.block_coords}]: filtered_point_data{point_data.dims}{point_data.shape}:  "
                  f"range=[{np.nanmin(point_data.values):.4f}, {np.nanmax(point_data.values):.4f}], nnan={nnan(point_data)}")

        result: xa.DataArray = tm().norm(point_data) if norm else point_data
        return result.astype( base_raster.dtype )

    # def raster2points2(self, base_raster: xa.DataArray, **kwargs) -> Tuple[Optional[xa.DataArray], Optional[np.ndarray],  Optional[np.ndarray]]:  # base_raster dims: [ band, y, x ]
    #     if (base_raster is None) or (base_raster.shape[0] == 0): return (None, None, None)
    #     point_data = base_raster.stack(samples=base_raster.dims[-2:]).transpose()
    #
    #     if '_FillValue' in point_data.attrs:
    #         nodata = point_data.attrs.get('_FillValue',np.nan)
    #         point_data = point_data if np.isnan(nodata) else point_data.where(point_data != nodata, np.nan)
    #
    #     pvcnts = [ nnan( point_data.values[ic] ) for ic in range( point_data.shape[0] ) ]
    #     pmask: np.ndarray = ( np.array(pvcnts) < point_data.shape[1]*0.5 )
    #     lgm().log(f"#FPD[{self.block_coords}]: pmask shp={shp(pmask)}, nvalid={np.count_nonzero(pmask)})  ")
    #
    #     filtered_point_data: xa.DataArray = point_data[pmask,:]
    #     lgm().log(f"#FPD[{self.block_coords}]: band-filtered point_data shape = {filtered_point_data.shape}, nnan={nnan(filtered_point_data)} ")
    #     smean: np.ndarray = np.nanmean( filtered_point_data.values, axis=0 )
    #     for iB in range( smean.size ):
    #         bmask: np.ndarray = np.isnan( filtered_point_data.values[:,iB] )
    #         filtered_point_data[ bmask, iB ] = smean[iB]
    #
    #     point_index = np.arange(0, base_raster.shape[-1] * base_raster.shape[-2])
    #     filtered_point_data.attrs['dsid'] = base_raster.name
    #
    #     lgm().log(f"#FPD[{self.block_coords}]: filtered_point_data{filtered_point_data.dims}{filtered_point_data.shape}:  "
    #               f"range=[{np.nanmin(filtered_point_data.values):.4f}, {np.nanmax(filtered_point_data.values):.4f}], nnan={nnan(filtered_point_data)}")
    #
    #     return filtered_point_data.assign_coords( samples=point_index[pmask]), pmask, pmask.reshape(base_raster.shape[1:])

#     def raster2points1( self, base_raster: xa.DataArray ) -> Tuple[ Optional[xa.DataArray], Optional[np.ndarray], Optional[np.ndarray] ]:   #  base_raster dims: [ band, y, x ]
#         t0 = time.time()
#         if base_raster is None: return (None, None, None)
#         rmask = ~np.isnan( base_raster.values.max(axis=0).squeeze() ) if (base_raster.size > 0) else None
# #        lgm().log( f"raster2points: stack spatial dims: {base_raster.dims[-2:]} (last dim varies fastest)" )
#         point_data = base_raster.stack(samples=base_raster.dims[-2:]).transpose()
#         if '_FillValue' in point_data.attrs:
#             nodata = point_data.attrs['_FillValue']
#             point_data = point_data if np.isnan( nodata ) else point_data.where( point_data != nodata, np.nan )
#         pmask: np.ndarray = ~np.isnan(point_data.values)
#         if pmask.ndim == 2: pmask = pmask.any(axis=1)
#   #      if self._point_coords is not None: pmask = self.mask & pmask
#         point_index = np.arange( 0, base_raster.shape[-1]*base_raster.shape[-2] )
#         filtered_point_data: xa.DataArray = point_data[ pmask, : ] if ( point_data.ndim == 2 ) else point_data[ pmask ]
#         filtered_point_data.attrs['dsid'] = base_raster.name
#         rnonz, pnonz = nz(rmask), nz(pmask)
#         lgm().log( f"raster2points -> [{base_raster.name}]: filtered_point_data shape = {filtered_point_data.shape}, "
#                    f"range=[{filtered_point_data.values.min():.4f}, {filtered_point_data.values.max():.4f}]" )
#         lgm().log( f"#IA: raster2points:  base_raster{base_raster.dims} shp={base_raster.shape}, "
#                    f" rmask shp,nz= ({shp(rmask)},{rnonz}), pmask shp,nz= ({shp(pmask)},{pnonz})  ")
#         lgm().log( f" ---> mask shape = {pmask.shape}, mask #valid = {np.count_nonzero(pmask)}/{pmask.size}, completed in {time.time()-t0} sec" )
#         lgm().log( f"filtered_point_data{filtered_point_data.dims}{filtered_point_data.shape} samples: \n ")
#         for iS in range( 100 ):
#            for iB in range(100):
#                 if np.isnan( filtered_point_data.values[iS,iB] ):
#                     print( f"  * NANVAL: S={iS} B={iB}")
#         return filtered_point_data.assign_coords( samples=point_index[ pmask ] ), pmask, rmask

    def coords2gid(self, cy, cx) -> Tuple[int,int,int]:
        index = self.coords2indices(cy, cx)
        ix, iy = index['ix'], index['iy']
        gid = ix + self.shape[-1] * iy
        return gid,ix,iy

#        from spectraclass.gui.control import UserFeedbackManager, ufm
#         try:
#             pdata, pcoords = self.getPointData()
#             s, x, y = pdata.samples.values, self.data.x.values, self.data.y.values
#             x0,y0 = x[0],y[0]
#             pids = np.where( s == gid )
#             spid = -1 if len(pids) == 0 else pids[0]
#             ufm().show( f"(ix,iy) => ({ix},{iy}) -> {gid}, spid={spid}, dx={int(x[ix]-x0)}, dy={int(y[iy]-y0)}, (cx,cy)=({int(cx)},{int(cy)})")
#             return gid
# #            return self.index_array.values[ index['iy'], index['ix'] ]
#         except IndexError as err:
#             ufm().show( f"(ix,iy) => ({ix},{iy}) -> {gid}: Exception: {err}" )
#             lgm().trace( f"coords2pindex ERROR: {err}")
#             return -1

    def multi_coords2gid(self, ycoords: List[float], xcoords: List[float]) -> np.ndarray:
        ( yi, xi ) = self.multi_coords2indices( ycoords, xcoords )
        return xi + self.shape[1] * yi


