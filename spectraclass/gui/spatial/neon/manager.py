import os.path

from spectraclass.gui.control import UserFeedbackManager, ufm
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from holoviews.streams import SingleTap, DoubleTap
from spectraclass.data.spatial.tile.tile import Block
from panel.layout.base import Panel
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.model.labels import LabelsManager, lm
import pandas as pd
from spectraclass.data.spatial.satellite import spm
from typing import List, Union, Tuple, Optional, Dict, Callable
import panel as pn
import time, math, sys, xarray as xa
from spectraclass.data.modes import BlockSelectMode
import holoviews as hv
from holoviews import opts
from holoviews import streams


class NEONTileSelector:

    @log_timing
    def __init__(self, **kwargs):
        self.selection_mode: BlockSelectMode = kwargs.get('mode',BlockSelectMode.LoadTile)
        self.selection_boxes = hv.Rectangles([]).opts( active_tools=['box_edit'], fill_alpha=0.75 )
        self.box_selection = streams.BoxEdit( source=self.selection_boxes, num_objects=1 )
        if self.selection_mode == BlockSelectMode.LoadTile: self.tap_stream = DoubleTap( transient=True )
        else:                                               self.tap_stream = SingleTap( transient=True )
        self.selected_rec = hv.DynamicMap(self.select_rec, streams=[self.tap_stream])
        self.rectangles: hv.Rectangles = None # ([(0, 0, 1, 1), (2, 3, 4, 6), (0.5, 2, 1.5, 4), (2, 1, 3.5, 2.5)])
        self.selected_rectangles: Dict[Tuple,Tuple] = {}
        self.rect_grid: Dict[Tuple,Tuple] = {}
        self.xlim, self.ylim = (sys.float_info.max, -sys.float_info.max), (sys.float_info.max, -sys.float_info.max)
        self.bdx, self.bdy = None, None
        self.bx0, self.by1 = None, None
        self._select_all = pn.widgets.Button( name='Select All', button_type='primary', width=150 )
        self._select_all.on_click( self.select_all )
        self._select_region = pn.widgets.Button( name='Select Region', button_type='primary', width=150 )
        self._select_region.on_click( self.select_region )
        self._clear_all  = pn.widgets.Button( name='Clear All',  button_type='warning', width=150 )
        self._clear_all.on_click( self.clear_all )
        self._clear_region  = pn.widgets.Button( name='Clear Region',  button_type='warning', width=150 )
        self._clear_region.on_click( self.clear_region )
        self.selection_name = pn.widgets.TextInput(name='Selection Name', placeholder='Give this selection a name...')
        self.load_button = pn.widgets.Button( name='Load Selection',  button_type='success', width=150 )
        self.load_button.on_click( self.load_selection )
        self.save_button = pn.widgets.Button( name='Save Selection',  button_type='success', width=150 )
        self.save_button.on_click( self.save_selection )


    def save_selection(self, event):
        sname = self.selection_name.value
        if sname:
            rect_indices = np.array(list(self.selected_rectangles.keys()))
            ufm().show(f"Save selection: {sname}, shape={rect_indices.shape}")
            pdata = pd.DataFrame( rect_indices, columns=['x','y'] )
            pdata.to_csv( os.path.expanduser("~/save_selection.csv") )


    def load_selection(self, event):
        sname = self.selection_name.value
        if sname:
            ufm().show(f"Load selection: {sname}")

    def get_save_panel(self) -> Panel:
        return pn.Column( self.selection_name, self.save_button )

    def get_blocks_in_region(self, bounds: Dict ) -> List[Tuple]:
        blocks = []
        [bx0,by0,bx1,by1] = [ bounds[k][0] for k in ('x0','y0','x1','y1') ]
        for bid, (x0,y0,x1,y1) in self.rect_grid.items():
            ufm().show(f"get_blocks_in_region: {x0} <-> {bx0}")
            if (bx0 < x1) and (bx1 > x0) and (by0 < y1) and (by1 > y0):
                blocks.append(bid)
        return blocks

    def update(self):
        self.selected_rec.event(x=None, y=None)

    def select_all(self, event ):
        self.selected_rectangles = self.rect_grid.copy()
        self.update()


    def select_region(self, event ):
        for bid in self.get_blocks_in_region( self.box_selection.data ):
            self.selected_rectangles[bid] = self.rect_grid[bid]
        self.update()

    def clear_all(self, event ):
        self.selected_rectangles = {}
        self.update()

    def clear_region(self, event ):
        for bid in self.get_blocks_in_region( self.box_selection.data ):
            self.selected_rectangles.pop( bid, None )
        self.update()

    def get_load_panel(self):
        return pn.Column([])

    def get_selection_panel(self):
        control_buttons = pn.Row( self._select_all, self._select_region, self._clear_all, self._clear_region )
        return pn.Column( control_buttons, self.get_save_panel() )

    @exception_handled
    def select_rec(self, x, y ):
        if x is not None:
            bindex =self.block_index(x,y)
            try:
                new_rect = self.rect_grid[bindex]
            except KeyError as err:
                lgm().log( f"Error accessing block, bindex={bindex}, rect keys={list(self.rect_grid.keys())}")
                raise err

            lgm().log(f"NTS: NEONTileSelector-> select block {bindex}, new_rect={new_rect}" )
            if bindex not in self.selected_rectangles:
                ufm().show( f"select block {bindex}")

                if self.selection_mode == BlockSelectMode.LoadTile:
                    tm().setBlock(bindex)
                    ufm().clear()
                    self.selected_rectangles = { bindex: new_rect }
                else:
                    self.selected_rectangles[bindex] = new_rect

            elif self.selection_mode == BlockSelectMode.SelectTile:
                ufm().show(f"clear block {bindex}")
                self.selected_rectangles.pop( bindex )

        return hv.Rectangles( self.selected_rectangles.values() ).opts( line_color="white", fill_alpha=0.5, line_alpha=1.0, line_width=1 )

    def gui(self):
        blocks: List[Block] = tm().tile.getBlocks()
        for block in blocks:
            (bxlim, bylim) = block.get_extent( spm().projection )
            self.xlim = ( min(bxlim[0],self.xlim[0]), max(bxlim[1],self.xlim[1]) )
            self.ylim = ( min(bylim[0],self.ylim[0]), max(bylim[1],self.ylim[1]) )
            dx, dy = (bxlim[1]-bxlim[0]), (bylim[1]-bylim[0])
            lgm().log( f"TS: BLOCK{block.block_coords}: dx={dx:.1f}, dy={dy:.1f}")
            if self.bdx is None:
                self.bdx, self.bdy = dx, dy
        self.bx0, self.by1 = self.xlim[0], self.ylim[1]
        lgm().log(f"TS:  x0={self.bx0:.1f}, y0={self.by1:.1f}, dx={self.bdx:.1f}, dy={self.bdy:.1f}")
        for block in blocks:
            (bxlim, bylim) = block.get_extent( spm().projection )
            r = (bxlim[0],bylim[0],bxlim[1],bylim[1])
            self.rect_grid[ block.block_coords] =  r
        lgm().log( f"TS: nblocks={len(blocks)}, nindices={len(self.rect_grid)}, indices={list(self.rect_grid.keys())}")
        self.rect0 = tm().block_index
        basemap = spm().get_image_basemap( self.xlim + self.ylim )
        self.rectangles = hv.Rectangles(list(self.rect_grid.values())).opts(line_color="cyan", fill_alpha=0.0, line_alpha=1.0)
        image = basemap * self.rectangles * self.selected_rec
        if self.selection_mode == BlockSelectMode.LoadTile:
            return image
        else:
            region_selection = self.selection_boxes.opts(
                opts.Rectangles(active_tools=['box_edit'], fill_alpha=0.5, line_alpha=1.0, line_color="white", fill_color="white"))
            selection_panel = self.get_selection_panel()
            return pn.Column( image * region_selection, selection_panel )

    @exception_handled
    def block_index(self, x, y ) -> Tuple[int,int]:
        if x is None: (x,y) = tm().block_index
        if type(x) == int: return (x,y)
        bindex =   math.floor( (x-self.bx0)/self.bdx ),  math.floor( (self.by1-y)/self.bdy )
        lgm().log( f"TS: block_index: {bindex}, x,y={(x,y)}, bx0={self.bx0}, by1={self.by1}, bdx,bdy={(self.bdx,self.bdy)}" )
        return bindex

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
        lgm().log( f"NEONDatasetManager: Found {self.nimages} images "  )


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

