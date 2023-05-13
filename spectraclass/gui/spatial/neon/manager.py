from spectraclass.gui.control import UserFeedbackManager, ufm
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from cartopy.mpl.geoaxes import GeoAxes
from spectraclass.xext.xgeo import XGeo
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.satellite import spm
from typing import List, Union, Tuple, Optional, Dict, Callable
import panel as pn, holoviews as hv
import time, math, sys, xarray as xa

class NEONTileSelector:

    @log_timing
    def __init__(self, **kwargs):
        self.init_band = kwargs.get( "init_band", 160 )
        self.grid_color = kwargs.get("grid_color", 'white')
        self.selection_color = kwargs.get("selection_color", 'black')
        self.slw = kwargs.get("slw", 2)
        self.colorstretch = 2.0
 #       self._blocks: Dict[Tuple[int,int],Rectangle] = {}
        self. rectangles: hv.Rectangles = None # ([(0, 0, 1, 1), (2, 3, 4, 6), (0.5, 2, 1.5, 4), (2, 1, 3.5, 2.5)])
        self._transformed_block_data = None
        self._selected_block: Tuple[int,int] = (0,0)

        self._band_index = 0
        self._select_rec = None

    def gui(self):
        blocks: List[Block] = tm().tile.getBlocks()
        rects = []
        xlim,ylim = (sys.float_info.max,-sys.float_info.max), (sys.float_info.max,-sys.float_info.max)
        for block in blocks:
            (bxlim, bylim) = block.get_extent( spm().projection )
            rects.append( (bxlim[0],bylim[0],bxlim[1],bylim[1]) )
            xlim = ( min(bxlim[0],xlim[0]), max(bxlim[1],xlim[1]) )
            ylim = ( min(bylim[0],ylim[0]), max(bylim[1],ylim[1]) )
        basemap = spm().selection_basemap(xlim,ylim)
        self.rectangles = hv.Rectangles( rects )
        return basemap * self.rectangles.opts( line_color="magenta", fill_alpha=0.0, line_alpha=1.0 )

    @property
    def image_index(self) -> int:
        from spectraclass.data.base import DataManager, dm
        return dm().modal.file_selector.index

    @property
    def image_name(self) -> str:
        from spectraclass.data.base import DataManager, dm
        return dm().modal.get_image_name( self.image_index )

    def get_color_bounds( self, raster: xa.DataArray ):
        ave = np.nanmean( raster.values )
        std = np.nanstd(  raster.values )
        nan_mask = np.isnan( raster.values )
        nnan = np.count_nonzero( nan_mask )
        lgm().log( f" **get_color_bounds: mean={ave}, std={std}, #nan={nnan}" )
        return dict( vmin= ave - std * self.colorstretch, vmax= ave + std * self.colorstretch  )

class NEONDatasetManager:

    @log_timing
    def __init__(self, **kwargs):
        from spectraclass.data.base import DataManager
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        self.dm: DataManager = DataManager.initialize( "DatasetManager", 'aviris' )
        if "cache_dir" in kwargs: self.dm.modal.cache_dir = kwargs["cache_dir"]
        if "data_dir"  in kwargs: self.dm.modal.data_dir = kwargs["data_dir"]
        if "images_glob" in kwargs: self.dm.modal.images_glob = kwargs["images_glob"]
        self.init_band = kwargs.get( "init_band", 160 )
        self.grid_color = kwargs.get("grid_color", 'white')
        self.selection_color = kwargs.get("selection_color", 'black')
        self.slw = kwargs.get("slw", 3)
        self.colorstretch = 2.0
        self.dm.proc_type = "cpu"
        self._selected_block: Tuple[int,int] = (0,0)
        TileManager.block_size = kwargs.get( 'block_size',  250 )
        self.nimages = len( self.dm.modal.image_names )
        self._nbands = None
        lgm().log( f"AvirisDatasetManager: Found {self.nimages} images "  )


    @property
    def image_index(self) -> int:
        return self.dm.modal.file_selector.index

    @property
    def image_name(self) -> str:
        return self.dm.modal.get_image_name( self.image_index )

    def clear_block_cache(self):
        self._transformed_block_data = None

    def get_color_bounds( self, raster: xa.DataArray ):
        ave = np.nanmean( raster.values )
        std = np.nanstd(  raster.values )
        nan_mask = np.isnan( raster.values )
        nnan = np.count_nonzero( nan_mask )
        lgm().log( f" **get_color_bounds: mean={ave}, std={std}, #nan={nnan}" )
        return dict( vmin= ave - std * self.colorstretch, vmax= ave + std * self.colorstretch  )

    def get_data_block(self, coords: Tuple[int,int] = None ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if coords is None: coords = self._selected_block
        data_array: xa.DataArray = tm().tile.data
        bsize = tm().block_size
        ix, iy = coords[0] * bsize, coords[1] * bsize
        dblock = data_array[ :, iy:iy+bsize, ix:ix+bsize ]
        t0 = dblock.attrs['transform']
        dblock.attrs['transform'] = [ t0[0], t0[1], t0[2] + ix * t0[0] + iy * t0[1], t0[3], t0[4], t0[5] + ix * t0[3] + iy * t0[4] ]
        return self.mask_nodata( dblock )

    def mask_nodata(self, data_array: xa.DataArray ) -> xa.DataArray:
        nodata = data_array.attrs.get('_FillValue')
        if nodata is not None:
            nodata_mask: np.ndarray = (data_array.values == nodata).flatten()
            nnan = np.count_nonzero(nodata_mask)
            flattened_data = data_array.values.flatten()
            flattened_data[nodata_mask] = np.nan
            data_array = data_array.copy( data=flattened_data.reshape(data_array.shape) )
            print(f" $$$$ mask_nodata, shape={data_array.shape}, size={data_array.size}, #nan={nnan}, %nan={(nnan*100.0)/data_array.size:.1f}%")
        return data_array

