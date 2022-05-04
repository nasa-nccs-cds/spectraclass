import traceback

from skimage.transform import ProjectiveTransform
import numpy as np
from os import path
import numpy.ma as ma
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from pyproj import Transformer
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict
import os, math, pickle, time

def combine_masks( mask1: Optional[np.ndarray], mask2: Optional[np.ndarray] ) -> Optional[np.ndarray]:
    if mask1 is None: return mask2
    if mask2 is None: return mask1
    return mask1 & mask2

def shp( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return "NONE" if (array is None) else array.shape

def nz( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return "NONE" if array is None else np.count_nonzero(array)

def nnan( array: Optional[Union[np.ndarray,xa.DataArray]] ):
    return "NONE" if array is None else np.count_nonzero( ~np.isnan(array) )

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
    def data(self) -> xa.DataArray:
        if self._data is None:
            self._data = self._get_data()
            lgm().log( f"IA: block data, shape: {shp(self._data)}, dims: {self._data.dims}")
        return self._data

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

    def __init__(self, tile_index: int, **kwargs ):
        super(Tile, self).__init__(**kwargs)
        self._blocks = {}
        self._index = tile_index
        self._metadata: Dict = None
        self.subsampling: int =  kwargs.get('subsample',1)

    @property
    def metadata(self) -> Dict:
        if self._metadata is None:
            self.init_metadata()
        return self._metadata

    def block_nvalid(self, block: "Block" ) -> int:
        return int( self.metadata[ self.bsizekey( block.block_coords ) ] )

    def _get_data(self) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager
        return TileManager.instance().getTileData()

    @property
    def name(self) -> str:
        return self.data.attrs['tilename']

    def getDataBlock(self, ix: int, iy: int, **kwargs ) -> Optional["Block"]:
        if (ix,iy) in self._blocks: return self._blocks[ (ix,iy) ]
        return self._blocks.setdefault( (ix,iy), Block( self, ix, iy, self._index, **kwargs ) )

    @log_timing
    def getBlocks(self, **kwargs ) -> List["Block"]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        lgm().log( f"getBlocks: tile_shape[x,y]={tm().tile_shape},  block_dims[x,y]={tm().block_dims}, raster_shape={tm().tile.data.shape}")
        return [ self.getDataBlock( ix, iy, **kwargs ) for ix in range(0,tm().block_dims[0]) for iy in range(0,tm().block_dims[1]) ]

    def init_metadata(self):
        from spectraclass.data.base import DataManager, dm
        if not dm().hasMetadata(): self.saveMetadata()
        self._metadata = self.loadMetadata()

    def loadMetadata(self) -> Dict:
        from spectraclass.data.base import DataManager, dm
        file_path = dm().metadata_file
        mdata = {}
        try:
            with open( file_path, "r" ) as mdfile:
                print(f"Loading metadata from file: {file_path}")
                block_sizes = {}  # { (1,1): 244284, (0,0): 134321 }
                for line in mdfile.readlines():
                    try:
                        toks = line.split("=")
                        if toks[0].startswith('block_size'):
                            bstok = toks[0].split("-")
                            block_sizes[ (int(bstok[1]), int(bstok[2])) ] = int( toks[1] )
                        else:
                            mdata[toks[0]] = "=".join(toks[1:])
                    except Exception as err:
                        lgm().log( f"LoadMetadata: Error '{err}' reading line '{line}'" )
                mdata[ 'block_size' ] = block_sizes
        except Exception as err:
            lgm().log( f"Warning: can't read config file '{file_path}': {err}\n")
        return mdata

    @log_timing
    def band_data(self, iband: int ) -> np.ndarray:
        return self.data[ iband, :, : ].to_numpy().squeeze()

    @log_timing
    def block_slice_data(self, iband: int, xbounds: Tuple[int,int], ybounds: Tuple[int,int] ) -> np.ndarray:
        return self.data[ iband, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ].to_numpy().squeeze()

    @log_timing
    def saveMetadata_preread( self ):
        from spectraclass.data.base import DataManager, dm
        file_path = dm().metadata_file
        block_data: Dict[Tuple,int] = {}
        blocks: List["Block"] = self.getBlocks()
        lgm().log(f"------------ saveMetadata: raster shape = {self.data.shape}" )
        raster_band: np.ndarray = self.band_data( 0 )
        nodata = self.data.attrs.get('_FillValue')
        for block in blocks:
            xbounds, ybounds = block.getBounds()
            raster_slice: np.ndarray = raster_band[ ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
            lgm().log( f"   * READ raster slice[{block.block_coords}], xbounds={xbounds}, ybounds={ybounds}, slice shape={raster_slice.shape}, nodata val = {nodata}")
            if not np.isnan(nodata):
                raster_slice[ raster_slice == nodata ] = np.nan
            nodata_mask = np.isnan( raster_slice )
            block_data[ block.block_coords ] = np.count_nonzero(nodata_mask)

        os.makedirs( os.path.dirname(file_path), exist_ok=True )
        try:
            with open( file_path, "w" ) as mdfile:
                for (k,v) in self.data.attrs.items():
                    mdfile.write( f"{k}={v}\n" )
                for bcoords, bsize in block_data.items():
                    mdfile.write( f"{self.bsizekey(bcoords)}={bsize}\n" )
            lgm().log(f" ---> Writing metadata file: {file_path}", print=True)
        except Exception as err:
            lgm().log(f" ---> ERROR Writing metadata file at {file_path}: {err}", print=True)
            if os.path.isfile(file_path): os.remove(file_path)

    @log_timing
    def saveMetadata( self ):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        t0 = time.time()
        file_path = dm().metadata_file
        if tm().reprocess or not os.path.isfile(file_path):
            block_data: Dict[Tuple,int] = {}
            blocks: List["Block"] = self.getBlocks()
            print( f"Generating metadata for image {file_path}" )
            lgm().log(f"------------ saveMetadata: raster shape = {self.data.shape}" )
            nodata = self.data.attrs.get('_FillValue')
            for block in blocks:
                xbounds, ybounds = block.getBounds()
                try:
                    raster_slice: np.ndarray = tm().tile.data[ 0, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ].to_numpy().squeeze()
                    if not np.isnan(nodata):
                        raster_slice[ raster_slice == nodata ] = np.nan
                    valid_mask = ~np.isnan( raster_slice )
                    block_data[ block.block_coords ] = np.count_nonzero(valid_mask)
                except Exception as err:
                    lgm().log( f"Error processing block{block.block_coords}: xbounds={xbounds}, ybounds={ybounds}, base shape = {tm().tile.data.shape}")

            os.makedirs( os.path.dirname(file_path), exist_ok=True )
            try:
                with open( file_path, "w" ) as mdfile:
                    for (k,v) in self.data.attrs.items():
                        mdfile.write( f"{k}={v}\n" )
                    for bcoords, bsize in block_data.items():
                        mdfile.write( f"{self.bsizekey(bcoords)}={bsize}\n" )
                lgm().log(f" ---> Writing metadata file: {file_path}, time = {(time.time()-t0)/60} min", print=True)
            except Exception as err:
                lgm().log(f" ---> ERROR Writing metadata file at {file_path}: {err}", print=True)
                if os.path.isfile(file_path): os.remove(file_path)

    @log_timing
    def saveMetadata_parallel( self ):
        from multiprocessing import cpu_count, get_context, Pool
        from spectraclass.data.base import DataManager, dm
        file_path = dm().metadata_file
        block_data: Dict[Tuple,int] = {}
        blocks: List["Block"] = self.getBlocks()
        lgm().log(f"------------ saveMetadata: raster shape = {self.data.shape}" )
        nodata = self.data.attrs.get('_FillValue')

#        with get_context("spawn").Pool(processes=nproc) as p:
#            image_tiles = p.map(image_processor, image_ids)

        for block in blocks:
            xbounds, ybounds = block.getBounds()
            raster_slice: np.ndarray = self.block_slice_data( 0, xbounds, ybounds )
            lgm().log( f"   * READ raster slice[{block.block_coords}], xbounds={xbounds}, ybounds={ybounds}, slice shape={raster_slice.shape}")
            if not np.isnan(nodata):
                raster_slice[ raster_slice == nodata ] = np.nan
            nodata_mask = np.isnan( raster_slice )
            block_data[ block.block_coords ] = np.count_nonzero(nodata_mask)

        os.makedirs( os.path.dirname(file_path), exist_ok=True )
        try:
            with open( file_path, "w" ) as mdfile:
                for (k,v) in self.data.attrs.items():
                    mdfile.write( f"{k}={v}\n" )
                for bcoords, bsize in block_data.items():
                    mdfile.write( f"{self.bsizekey(bcoords)}={bsize}\n" )
            lgm().log(f" ---> Writing metadata file: {file_path}", print=True)
        except Exception as err:
            lgm().log(f" ---> ERROR Writing metadata file at {file_path}: {err}", print=True)
            if os.path.isfile(file_path): os.remove(file_path)

    def bsizekey(self, bcoords: Tuple[int,int] ) -> str:
        return f"nvalid-{bcoords[0]}-{bcoords[1]}"

class ThresholdRecord:

    def __init__(self, fdata: xa.DataArray ):
        self._tmask: xa.DataArray = None
        self.thresholds: Tuple[float,float] = (0.0,1.0)
        self.fixed: bool = False
        self.needs_update = [ True, True ]
        self.fdata: xa.DataArray = fdata
        self._drange = None

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
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.data.base import DataManager, dm
        super(Block, self).__init__( data_projected=True, **kwargs )
        self.tile: Tile = tile
        self.init_task = None
        self._tmask: xa.DataArray = None
        self._model_data: xa.DataArray = None
        self.config = kwargs
        self._trecs: Tuple[ Dict[int,ThresholdRecord], Dict[int,ThresholdRecord] ] = ( {}, {} )
        self.block_coords = (ix,iy)
        self.tile_index = itile
 #       self.validate_parameters()
        self._index_array: xa.DataArray = None
        self._pid_array: np.ndarray = None
        self._flow = None
        self._samples_axis: Optional[xa.DataArray] = None
        self._point_data: Optional[xa.DataArray] = None
        self._point_coords: Optional[Dict[str,np.ndarray]] = None
        self._point_mask: Optional[np.ndarray] = None
        self._raster_mask: Optional[np.ndarray] = None
 #       lgm().log(f"CREATE Block: ix={ix}, iy={iy}, dfile={dm().modal.dataFile(block=self, index=self.tile_index)}")

    def set_thresholds(self, bUseModel: bool, iFrame: int, thresholds: Tuple[float,float] ) -> bool:
        trec: ThresholdRecord = self.threshold_record( bUseModel, iFrame )
        initialized = trec.is_empty()
        mask = trec.set_thresholds( thresholds )
        lgm().log(f"#IA: MASK[{iFrame}].set_thresholds---> nmasked pixels = {np.count_nonzero(mask)} ")
        self._tmask = None
        self._index_array = None
        self.tile.initialize()
        return initialized

    def threshold_record(self, model_data: bool, iFrame: int ) -> ThresholdRecord:
        from spectraclass.data.base import dm
        trecs: Dict[int,ThresholdRecord] = self._trecs[ int(model_data) ]
        if iFrame in trecs: return trecs[ iFrame ]
        fdata: xa.DataArray = self.points2raster( dm().getModelData() ) if model_data else self.data
        return trecs.setdefault( iFrame,  ThresholdRecord( fdata[iFrame] ) )

    def get_mask_list(self, current_frame = -1 ) -> Tuple[ List[str], str ]:
        mask_list, types, value = [], ["band", "model" ], None
        for ttype, trecs in zip(types,self._trecs):
            for iFrame, trec in trecs.items():
                if trec.tmask is not None:
                    mask_name = f"{ttype}:{iFrame}"
                    mask_list.append( mask_name )
                    if iFrame == current_frame:
                        value = mask_name
        return ( mask_list, value )

    @exception_handled
    def get_threshold_mask( self, raster=False, reduced=True ) -> np.ndarray:
        lgm().log( f"#IA: get_threshold_mask[B-{hex(id(self))}] (raster={raster}) ************************************** **************************************" )
        if self._tmask is None:
            lgm().log( f"#IA: ***************> ntrecs={[len(trecs.keys()) for trecs in self._trecs]}")
            ntmask = None
            for trecs in self._trecs:
                for iFrame, trec in trecs.items():
                    lgm().log( f"#IA: ***************> Merging Frame-{iFrame} Threshold Mask, shape={shp(trec.tmask)}, nz={nz(trec.tmask)}" )
                    if trec.tmask is not None:
                        ntmask = trec.tmask if (ntmask is None) else (ntmask | trec.tmask)
            if ntmask is not None:
                self._tmask = ~ntmask
                lgm().log( f"#IA: ***************> Merging [tmask, nz={nz(self._tmask)}] & [rmask, nz={nz(self.raster_mask)}]" )
                self._tmask = self._tmask & self.raster_mask
        if self._tmask is not None:
            lgm().log( f"#IA: ***************> MASK--> shape = {shp(self._tmask)}, nz={nz(self._tmask)}" )
            if not raster:
                ptmask = self._tmask.values.flatten()
                if reduced: ptmask = ptmask[self._point_mask]
                lgm().log( f"#IA: ***************> MASK--> ptmask.shape={shp(ptmask)}, nz={nz(ptmask)} " )
                return ptmask
            return self._tmask.values

    def classmap(self, default_value: int =0 ) -> xa.DataArray:
        return xa.full_like( self.data[0].squeeze(drop=True), default_value, dtype=np.int )

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
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        raw_raster: Optional[xa.DataArray] = self.load_block_raster()
        if raw_raster is None:
            if self.tile.data is None: return None
            xbounds, ybounds = self.getBounds()
            raster_slice = self.tile.data[:, ybounds[0]:ybounds[1], xbounds[0]:xbounds[1] ]
            raw_raster = raster_slice if (raster_slice.size == 0) else TileManager.process_tile_data( raster_slice )
            lgm().log(f"BLOCK{self.block_coords}: load-slice ybounds={ybounds}, xbounds={xbounds}, raster shape={raw_raster.shape}")
        block_raster = self._apply_mask( raw_raster )
        block_raster.attrs['block_coords'] = self.block_coords
        block_raster.attrs['tile_shape'] = tm().tile.data.shape
        block_raster.attrs['block_dims'] = tm().block_dims
        block_raster.attrs['tile_size']  = tm().tile_size
        block_raster.attrs['dsid'] = self.dsid()
        block_raster.attrs['file_name'] = self.file_name
        block_raster.name = self.file_name
        return block_raster

    @property
    def data_file(self):
        from spectraclass.data.base import DataManager, dm
        return dm().modal.dataFile( block=self, index=self.tile_index )

    @log_timing
    def has_data_samples(self) -> bool:
        file_exists = path.isfile(self.data_file)
        if file_exists:
            with xa.open_dataset(self.data_file) as dataset:
                nsamples = 0 if (len( dataset.coords ) == 0) else dataset.coords['samples'].size
                lgm().log( f" BLOCK{self.block_coords} data_samples={nsamples}")
                file_exists = (nsamples > 0)
        return file_exists

    def has_data_file(self, non_empty=False ) -> bool:
        file_exists = path.isfile(self.data_file)
        if non_empty and file_exists:
            return self.has_data_samples()
        return file_exists

    @log_timing
    def load_block_raster(self) -> Optional[xa.DataArray]:
        from spectraclass.data.base import DataManager, dm
        from spectraclass.gui.control import UserFeedbackManager, ufm
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        raw_raster: Optional[xa.DataArray] = None
        if self.has_data_file():
            dataset: xa.Dataset = dm().modal.loadDataFile( block=self, index=self.tile_index )
            raw_raster = tm().mask_nodata( dataset["raw"] )
            lgm().log( f" ---> load_block_raster{self.block_coords}: raw data attrs = {dataset['raw'].attrs.keys()}" )
            for aid, aval in dataset.attrs.items():
                if aid not in raw_raster.attrs:
                    raw_raster.attrs[aid] = aval
            lgm().log(f"BLOCK{self.block_coords}->get_data: load-datafile raster shape={raw_raster.shape}")
            if raw_raster.size == 0: ufm().show( "This block does not appear to have any data.", "red" )
        return raw_raster

    @property
    def model_data(self):
        if self._model_data is None:
            self._model_data = self._get_model_data()
        return self._model_data

    def _get_model_data(self) -> xa.DataArray:
        from spectraclass.data.base import DataManager, dm
        dataset: Optional[xa.Dataset] = dm().modal.loadDataFile( block=self, index=self.tile_index )
        point_data = dataset["reduction"]
        point_data.attrs['block_coords'] = self.block_coords
        point_data.attrs['dsid'] = self.dsid()
        point_data.attrs['file_name'] = self.file_name
        point_data.name = self.file_name
        return point_data

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
        return result

    def addTextureBands( self ):
        from spectraclass.features.texture.manager import TextureManager, texm
        self._data = texm().addTextureBands( self.data )

    @property
    def file_name(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return f"{tm().tileName()}_b-{tm().block_size}-{self.block_coords[0]}-{self.block_coords[1]}"

    def get_pid_array(self) -> np.ndarray:
        d0: np.ndarray = self.data.values[0].squeeze().flatten()
        pids = np.arange(d0.size)
        return pids[ ~np.isnan(d0) ]

    @property
    def shape(self) -> Tuple[int,...]:
        return self.data.shape

    @property
    def zeros(self) -> np.ndarray:
        return np.zeros( self.shape, np.int)

    def getBounds(self ) -> Tuple[ Tuple[int,int], Tuple[int,int] ]:
        from spectraclass.data.spatial.tile.manager import tm
        bsize = tm().block_size
        x0, y0 = self.block_coords[0]*bsize, self.block_coords[1]*bsize
        bounds = ( x0, x0+bsize ), ( y0, y0+bsize )
        lgm().log( f"GET BLOCK{self.block_coords} BOUNDS: dx={bounds[0]}, dy={bounds[1]}")
        return bounds

    @log_timing
    def getPointData( self ) -> Tuple[xa.DataArray,Dict]:
        if self._point_data is None:
            self._point_data, pmask, rmask =  self.raster2points( self.data )
            self._point_coords: Dict[str,np.ndarray] = dict( y=self.data.y.values, x=self.data.x.values, mask=pmask )
            self._samples_axis = self._point_data.coords['samples']
            self._point_data.attrs['type'] = 'block'
            self._point_data.attrs['dsid'] = self.dsid()
            self._point_mask = pmask
            self._raster_mask = rmask
        return (self._point_data, self._point_coords )

    @property
    def point_mask(self) -> np.ndarray:
        if self._point_mask is None: self.getPointData()
        return self._point_mask

    @property
    def raster_mask(self) -> np.ndarray:
        if self._raster_mask is None: self.getPointData()
        return self._raster_mask

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

    def points2raster(self, points_data: xa.DataArray ) -> xa.DataArray:
        tmask = self.get_threshold_mask( reduced=False )
        lgm().log( f"points->raster, points: dims={points_data.dims}, shape={points_data.shape}; data: dims={self.data.dims}, shape={self.data.shape}")
        lgm().log( f" ---> pmask: shape = {self.mask.shape}, #nonzero = {np.count_nonzero(self.mask)}")
        dims = [points_data.dims[1], self.data.dims[1], self.data.dims[2]]
        coords = [(dims[0], points_data[dims[0]].data), (dims[1], self.data[dims[1]].data), (dims[2], self.data[dims[2]].data)]
        raster_data = np.full([self.data.shape[1] * self.data.shape[2], points_data.shape[1]], float('nan'))
        pmask = self.mask if (tmask is None) else tmask
        raster_data[ pmask ] = points_data.data
        raster_data = raster_data.transpose().reshape([points_data.shape[1], self.data.shape[1], self.data.shape[2]])
        lgm().log( f"Generated Raster data, shape={raster_data.shape}, dims={dims}, with mask shape={self.mask.shape}" )
        return xa.DataArray( raster_data, coords, dims, points_data.name, points_data.attrs )

    def raster2points( self, base_raster: xa.DataArray ) -> Tuple[ Optional[xa.DataArray], Optional[np.ndarray], Optional[np.ndarray] ]:   #  base_raster dims: [ band, y, x ]
        t0 = time.time()
        if base_raster is None: return (None, None, None)
        rmask = ~np.isnan( base_raster[0].values.squeeze() ) if (base_raster.size > 0) else None
        lgm().log( f"raster2points: stack spatial dims: {base_raster.dims[-2:]} (last dim varies fastest)" )
        point_data = base_raster.stack(samples=base_raster.dims[-2:]).transpose()
        if '_FillValue' in point_data.attrs:
            nodata = point_data.attrs['_FillValue']
            point_data = point_data if np.isnan( nodata ) else point_data.where( point_data != nodata, np.nan )
        pmask: np.ndarray = ~np.isnan(point_data.values) if (self._point_coords is None) else self.mask
        if pmask.ndim == 2: pmask = pmask.any(axis=1)
        point_index = np.arange( 0, base_raster.shape[-1]*base_raster.shape[-2] )
        filtered_point_data: xa.DataArray = point_data[ pmask, : ] if ( point_data.ndim == 2 ) else point_data[ pmask ]
        filtered_point_data.attrs['dsid'] = base_raster.name
        lgm().log( f"raster2points -> [{base_raster.name}]: filtered_point_data shape = {filtered_point_data.shape}" )
        lgm().log( f"#IA: raster2points:  base_raster{base_raster.dims} shp={base_raster.shape}, rmask shp={shp(rmask)}, nz={nz(rmask)} ")
        lgm().log( f" ---> mask shape = {pmask.shape}, mask #valid = {np.count_nonzero(pmask)}/{pmask.size}, completed in {time.time()-t0} sec" )
        return filtered_point_data.assign_coords( samples=point_index[ pmask ] ), pmask, rmask

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


