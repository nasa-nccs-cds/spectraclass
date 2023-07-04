import os.path

from spectraclass.gui.control import UserFeedbackManager, ufm
import param, numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from holoviews.streams import SingleTap, DoubleTap
from spectraclass.data.spatial.tile.tile import Block
from panel.layout.base import Panel
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.model.labels import LabelsManager, lm
import pandas as pd
from spectraclass.data.spatial.satellite import spm
from typing import List, Union, Tuple, Optional, Dict, Callable
import panel as pn, glob
import time, math, sys, xarray as xa
from spectraclass.data.modes import BlockSelectMode
from spectraclass.data.base import DataManager, dm
from spectraclass.model.base import SCSingletonConfigurable
import holoviews as hv
from holoviews import opts
from holoviews import streams

def nts(**kwargs): return NEONTileSelector.instance(**kwargs)

class BlockSelection(param.Parameterized):
    selection_name = param.String(default="", doc="Name of saved block selection")

    def __init__(self):
        super(BlockSelection, self).__init__()
        self._selected_rectangles: Dict[Tuple,Tuple] = {}
        self.selection_name_input = pn.widgets.TextInput(name='Selection Name', placeholder='Give this selection a name...')
        self.selection_name_input.link( self, value='selection_name' )
        self.rect_grid: Dict[Tuple,Tuple] = {}
        self.load_button = pn.widgets.Button( name='Load Selection',  button_type='success', width=150 )
        self.load_button.on_click( self.load_selection )
        self.save_button = pn.widgets.Button( name='Save Selection',  button_type='success', width=150 )
        self.save_button.on_click( self.save_selection )
        self.save_dir = f"{dm().cache_dir}/masks/block_selection"
        os.makedirs(self.save_dir, exist_ok=True)
        self.fill_rect_grid()

    def fill_rect_grid(self):
        blocks: List[Block] = tm().tile.getBlocks()
        for block in blocks:
            (bxlim, bylim) = block.get_extent(spm().projection)
            r = (bxlim[0], bylim[0], bxlim[1], bylim[1])
            self.rect_grid[block.block_coords] = r

    def get_blocks_in_region(self, bounds: Dict ) -> List[Tuple]:
        blocks = []
        [bx0,by0,bx1,by1] = [ bounds[k][0] for k in ('x0','y0','x1','y1') ]
        for bid, (x0,y0,x1,y1) in self.rect_grid.items():
            if (bx0 < x1) and (bx1 > x0) and (by0 < y1) and (by1 > y0):
                blocks.append(bid)
        return blocks

    def add_grid_rect(self, bid: Tuple , rect: Tuple ):
        self.rect_grid[bid] = rect

    def grid_widget(self) -> hv.Rectangles:
        return hv.Rectangles(list(self.rect_grid.values())).opts(line_color="cyan", fill_alpha=0.0, line_alpha=1.0)

    def select_all(self):
        self._selected_rectangles = self.rect_grid.copy()

    def clear_all(self):
        self._selected_rectangles = {}

    @exception_handled
    def select_block( self, bid: Tuple ):
        self._selected_rectangles[bid] = self.rect_grid[bid]

    @property
    def selected_bids(self) -> List[Tuple]:
        return list(self._selected_rectangles.keys())

    def block_selected(self, bid: Tuple) -> bool:
        return (bid in self._selected_rectangles)

    def clear_block( self, bid: Tuple ):
        self._selected_rectangles.pop(bid, None)

    @property
    def selected_rectangles(self) -> List[Tuple]:
        return list(self._selected_rectangles.values())

    @exception_handled
    def save_selection(self, event):
        sname = self.selection_name
        if sname:
            rect_indices = np.array(list(self._selected_rectangles.keys()))
            pdata = pd.DataFrame( rect_indices, columns=['x','y'] )
            save_file = f"{self.save_dir}/{tm().tileid}.{sname}.csv"
            ufm().show(f"Save selection: {sname}, shape={rect_indices.shape}, file='{save_file}'")
            try:
                pdata.to_csv( save_file )
            except Exception as err:
                ufm().show(f"Error saving file: {err}")

    def load_selection(self, event):
        sname = self.selection_name
        if sname:
            save_file = f"{self.save_dir}/{tm().tileid}.{sname}.csv"
            ufm().show(f"Load selection: {sname}, file='{save_file}'")
            try:
                pdata: pd.DataFrame = pd.read_csv( save_file )
                self._selected_rectangles = {}
                for index, row in pdata.iterrows():
                    bid = (row['x'],row['y'])
                    self._selected_rectangles[bid] = self.rect_grid[bid]
                self.update()
            except Exception as err:
                ufm().show(f"Error loading file: {err}")

    def get_selection_load_panel(self, event=None ):
        block_selection_names = [ f.split(".")[-2] for f in os.listdir(self.save_dir) ]
        file_selector = pn.widgets.Select( name='Tile Mask', options=block_selection_names, value=block_selection_names[0] )
        file_selector.link( self, value='selection_name' )
        return pn.Row( file_selector, self.load_button )

    def get_selection_save_panel(self, event=None ):
        return pn.Row(self.selection_name_input, self.save_button)

    def get_cache_panel(self) -> Panel:
        load_panel = self.get_selection_load_panel()
        save_panel = self.get_selection_save_panel()
        return  pn.Tabs( ("load",load_panel), ("save",save_panel) )

class NEONTileSelector(SCSingletonConfigurable):

    @log_timing
    def __init__(self, **kwargs):
        super(NEONTileSelector, self).__init__()
        self.selection_mode: BlockSelectMode = kwargs.get('mode',BlockSelectMode.LoadTile)
        self.selection_boxes: hv.Rectangles = hv.Rectangles([]).opts( active_tools=['box_edit'], fill_alpha=0.75 )
        self.box_selection = streams.BoxEdit( source=self.selection_boxes, num_objects=1 )
        self.region_selection: hv.Rectangles = self.selection_boxes.opts( opts.Rectangles(active_tools=['box_edit'], fill_alpha=0.75, line_alpha=1.0, line_color="yellow", fill_color="white"))
        if self.selection_mode == BlockSelectMode.LoadTile: self.tap_stream = DoubleTap( transient=True )
        else:                                               self.tap_stream = SingleTap( transient=True )
        self.selected_rec = hv.DynamicMap(self.select_rec, streams=[self.tap_stream])
        self.rectangles: hv.Rectangles = None # ([(0, 0, 1, 1), (2, 3, 4, 6), (0.5, 2, 1.5, 4), (2, 1, 3.5, 2.5)])
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
        self.blockSelection = BlockSelection()

    def update(self):
        self.selected_rec.event(x=None, y=None)

    def select_all(self, event ):
        ufm().show( "Select All")
        self.blockSelection.select_all()
        self.update()

    def select_region(self, event ):
        ufm().show("Select Region")
        for bid in self.blockSelection.get_blocks_in_region( self.box_selection.data ):
            self.blockSelection.select_block(bid)
        self.update()
        print( "=========>>>>> select_region: <<<<<=========>>>>> ")
        print( f"box_selection: {self.box_selection.data}" )
        print( f"selected_bids: {self.blockSelection.selected_bids}" )

    def clear_all(self, event ):
        ufm().show("Clear All")
        self.blockSelection.clear_all()
        self.update()

    def clear_region(self, event ):
        ufm().show("Clear Region")
        for bid in self.blockSelection.get_blocks_in_region( self.box_selection.data ):
            self.blockSelection.clear_block(bid)
        self.update()

    def get_load_panel(self):
        load_panel = self.blockSelection.get_selection_load_panel()
        return pn.Column(load_panel)

    def get_control_panel(self):
        select_buttons = pn.Row( self._select_all, self._select_region )
        clear_buttons = pn.Row( self._clear_all, self._clear_region)
        selection_panel = pn.Column( select_buttons, clear_buttons )
        cache_panel = self.blockSelection.get_cache_panel()
        control_panels = pn.Tabs( ("select",selection_panel), ("cache",cache_panel) )
        return control_panels

    @exception_handled
    def select_rec(self, x, y ):
 #       self.box_selection.data = {}    # Data can't be modified.
        if x is not None:
            bindex =self.block_index(x,y)
            lgm().log(f"NTS: NEONTileSelector-> select block {bindex}" )

            if not self.blockSelection.block_selected(bindex):
                ufm().show( f"select block {bindex}")

                if self.selection_mode == BlockSelectMode.LoadTile:
                    tm().setBlock(bindex)
                    ufm().clear()
                    self.blockSelection.clear()

                self.blockSelection.select_block(bindex)

            elif self.selection_mode == BlockSelectMode.SelectTile:
                ufm().show(f"clear block {bindex}")
                self.blockSelection.clear_block( bindex )

        return hv.Rectangles( self.blockSelection.selected_rectangles ).opts(line_color="white", fill_alpha=0.6, line_alpha=1.0, line_width=2)

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

        lgm().log( f"TS: nblocks={len(blocks)}, nindices={len(self.rect_grid)}, indices={list(self.rect_grid.keys())}")
        self.rect0 = tm().block_index
        basemap = spm().get_image_basemap( self.xlim + self.ylim )
        self.rectangles = self.blockSelection.grid_widget()
        image = basemap * self.rectangles * self.selected_rec
        if self.selection_mode == BlockSelectMode.LoadTile:
            return image
        else:
            selection_panel = self.get_control_panel()
            return pn.Row( image * self.region_selection, selection_panel )

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

