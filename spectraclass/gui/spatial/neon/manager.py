import os.path

from spectraclass.gui.control import UserFeedbackManager, ufm
import param, numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from holoviews.streams import SingleTap, DoubleTap
from spectraclass.data.spatial.tile.tile import Block
from panel.layout.base import Panel
from panel.pane import Alert
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.learn.cluster.manager import clm
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

class BlockSelection(param.Parameterized):
    selection_name = param.String(default="", doc="Name of saved block selection")

    def __init__(self, selection_mode: BlockSelectMode ):
        super(BlockSelection, self).__init__()
        self.selection_mode = selection_mode
        self.marker_color = "red"
        self.unmarked_color = "white"
        self._selected_rectangles: Dict[Tuple,Tuple] = None
        self.selection_name_input = pn.widgets.TextInput(name='Selection Name', placeholder='Give this selection a name...')
        self.selection_name_input.link( self, value='selection_name' )
        self.rect_grid: Dict[Tuple,Tuple] = {}
        self.load_button = pn.widgets.Button( name='Load Selection',  button_type='success', width=150 )
        self.load_button.on_click( self.load_selection )
        self.save_button = pn.widgets.Button( name='Save Selection',  button_type='success', width=150 )
        self.save_button.on_click( self.save_selection )
        self.click_select_mode = pn.widgets.RadioButtonGroup( name='Click-Select Mode', options=['Unselect', 'Mark'], value="Unselect", button_type='success')
        self.save_dir = f"{dm().cache_dir}/masks/block_selection"
        self.dynamic_selection = None
        os.makedirs(self.save_dir, exist_ok=True)
        self.fill_rect_grid()

    def get_dynamic_selection(self, streams ) -> hv.DynamicMap:
        self.dynamic_selection = hv.DynamicMap( self.select_rec, streams=streams )
        return self.dynamic_selection

    def update(self):
        self.dynamic_selection.event(x=None, y=None)

    @property
    def region_bounds(self):
        return self.xlim + self.ylim

    @exception_handled
    def block_index(self, x, y ) -> Tuple[int,int]:
        if x is None: (x,y) = tm().block_index
        if type(x) == int: return (x,y)
        bindex =   math.floor( (x-self.bx0)/self.bdx ),  math.floor( (self.by1-y)/self.bdy )
        lgm().log( f"TS: block_index: {bindex}, x,y={(x,y)}, bx0={self.bx0}, by1={self.by1}, bdx,bdy={(self.bdx,self.bdy)}" )
        return bindex
    @exception_handled
    def select_rec(self, x, y):
        #       self.box_selection.data = {}    # Data can't be modified.
        if x is not None:
            bindex = self.block_index(x, y)
            lgm().log(f"NTS: NEONTileSelector-> select block {bindex}")

            if not self.block_selected(bindex):
                ufm().show(f"select block {bindex}")

                if self.selection_mode == BlockSelectMode.LoadTile:
                    tm().setBlock(bindex)
                    ufm().clear()
                    self.clear_all()

                self.select_block(bindex)

            else:
                if self.click_select_mode.value == 'Unselect':
                    ufm().show(f"clear block {bindex}")
                    self.clear_block(bindex)
                else:
                    ufm().show(f"mark block {bindex}")
                    self.select_block(bindex)

        return hv.Rectangles(self.selected_rectangles, vdims = 'value').opts(color='value', fill_alpha=0.6, line_alpha=1.0, line_width=2)

    def fill_rect_grid(self):
        self.xlim, self.ylim = (sys.float_info.max, -sys.float_info.max), (sys.float_info.max, -sys.float_info.max)
        self.bdx, self.bdy = None, None
        self.bx0, self.by1 = None, None
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
        self._selected_rectangles = { k: v + (self.unmarked_color,) for k,v in self.rect_grid.items() }
        self.update()

    def clear_all(self):
        self._selected_rectangles = {}
        self.update()

    def clear_marker(self):
        if self._selected_rectangles is not None:
            for k in self._selected_rectangles.keys():
                if self._selected_rectangles[k][4] == self.marker_color:
                    self._selected_rectangles[k] = self._selected_rectangles[k][:4] + (self.unmarked_color,)

    @exception_handled
    def select_block( self, bid: Tuple, update=True ):
        if self._selected_rectangles is None:
            self._selected_rectangles = {}
        color = self.marker_color if update else self.unmarked_color
        self.clear_marker()
        try:
            self._selected_rectangles[bid] = self.rect_grid[bid] + (color,)
        except KeyError as err:
            lgm().log( f" KeyError in select_block, bid={bid}, rect keys={list(self.rect_grid.keys())}")
            raise err
        if update:
            tm().block_index = bid
            self.update()

    def select_region(self, bounds: Dict  ):
        ufm().show("Select Region")
        for bid in self.get_blocks_in_region( bounds ):
            self.select_block(bid,False)
        self.update()

    def clear_region(self, bounds: Dict  ):
        ufm().show("Clear Region")
        for bid in self.get_blocks_in_region( bounds ):
            self.clear_block(bid,False)
        self.update()

    @property
    def selected_bids(self) -> Optional[List[Tuple]]:
        if self._selected_rectangles is not None:
            return list(self._selected_rectangles.keys())

    def block_selected(self, bid: Tuple) -> bool:
        if self._selected_rectangles is None:  return False
        return (bid in self._selected_rectangles)

    def clear_block( self, bid: Tuple, update=True ):
        if self._selected_rectangles is not None:
            self._selected_rectangles.pop(bid, None)
            if update: self.update()

    @property
    def selected_rectangles(self) -> List[Tuple]:
        if self._selected_rectangles is None:  return []
        return list(self._selected_rectangles.values())

    @exception_handled
    def save_selection(self, *args ):
        if self._selected_rectangles is not None:
            sname = self.selection_name if self.selection_name else "block_selection"
            rect_indices = np.array(list(self._selected_rectangles.keys()))
            pdata = pd.DataFrame( rect_indices, columns=['x','y'] )
            save_file = f"{self.save_dir}/{tm().tileid}.{sname}.csv"
            ufm().show(f"Save block selection: {sname}")
            lgm().log(f" ----> file='{save_file}'")
            try:
                pdata.to_csv( save_file )
            except Exception as err:
                ufm().show(f"Error saving file: {err}")

    @exception_handled
    def load_selection(self, event):
        sname = self.selection_name
        if sname:
            save_file = f"{self.save_dir}/{tm().tileid}.{sname}.csv"
            ufm().show(f"Load Block mask '{sname}'")
            lgm().log(f"Load Block mask '{sname}': file='{save_file}'")
            dm().modal.update_parameter( "Block Mask", sname )
            pdata: pd.DataFrame = pd.read_csv( save_file )
            self._selected_rectangles = {}
            for index, row in pdata.iterrows():
                bid = (row['x'],row['y'])
                self._selected_rectangles[bid] = self.rect_grid[bid] + (self.unmarked_color,)
            self.update()

    def get_block_selecction(self) -> Optional[Dict]:
        return self._selected_rectangles

    def get_selection_load_panel(self):
        block_selection_names = [ f.split(".")[-2] for f in os.listdir(self.save_dir) ]
        sopts = dict( name='Block Mask', options=block_selection_names )
        if len(block_selection_names) > 0:
            sopts['value'] = self.selection_name = block_selection_names[0]
        file_selector = pn.widgets.Select( **sopts )
        file_selector.link( self, value='selection_name' )
        return pn.Row( file_selector, self.load_button  )

    def get_selection_save_panel(self, event=None ):
        return pn.Row(self.selection_name_input, self.save_button)

    def get_cache_panel(self, mode: BlockSelectMode) -> Panel:
        from spectraclass.learn.pytorch.trainer import mpt
        tabs = [ ("blocks", self.get_selection_load_panel()) ]
        panels= []
        title = "### Load Block Mask"
        if mode == BlockSelectMode.CreateMask:
            tabs.append( ("save", self.get_selection_save_panel()) )
            title = "### Save Block Mask"
        elif mode == BlockSelectMode.LoadMask:
            tabs.append( ("clusters", mpt().get_mask_load_panel()) )
            panels.append( ufm().gui() )
        return  pn.WidgetBox( title, *panels, pn.Tabs( *tabs ) )

class NEONTileSelector(param.Parameterized):

    @log_timing
    def __init__(self, **kwargs):
        super(NEONTileSelector, self).__init__()
        self.selection_mode: BlockSelectMode = kwargs.get('mode',BlockSelectMode.LoadTile)
        lgm().log( f"#NTS: selection_mode = {self.selection_mode}")
        self.region_selection: hv.Rectangles = hv.Rectangles([]).opts( active_tools=['box_edit'], fill_alpha=0.75, color="white" )
        self.box_selection = streams.BoxEdit( source=self.region_selection, num_objects=1 )
        self.blockSelection = BlockSelection( self.selection_mode )
        if self.selection_mode == BlockSelectMode.LoadTile: self.tap_stream = DoubleTap( transient=True )
        else:                                               self.tap_stream = SingleTap( transient=True )
        self.selected_rec = self.blockSelection.get_dynamic_selection( [self.tap_stream] )
        self.rect_grid: hv.Rectangles = None
        self._select_all = pn.widgets.Button( name='Select All', button_type='primary', width=150 )
        self._select_all.on_click( self.select_all )
        self._select_region = pn.widgets.Button( name='Select Region', button_type='primary', width=150 )
        self._select_region.on_click( self.select_region )
        self._clear_all  = pn.widgets.Button( name='Clear All',  button_type='warning', width=150 )
        self._clear_all.on_click( self.clear_all )
        self._clear_region  = pn.widgets.Button( name='Clear Region',  button_type='warning', width=150 )
        self._clear_region.on_click( self.clear_region )

    def save_block_selection(self):
        self.blockSelection.save_selection()

    def select_all(self, event ):
        ufm().show( "Select All")
        self.blockSelection.select_all()

    def select_region(self, event ):
        ufm().show("Select Region")
        self.blockSelection.select_region( self.box_selection.data )

    def clear_all(self, event ):
        ufm().show("Clear All")
        self.blockSelection.clear_all()

    def clear_region(self, event ):
        ufm().show("Clear Region")
        self.blockSelection.clear_region(self.box_selection.data)

    def get_load_panel(self):
        load_panel = self.blockSelection.get_selection_load_panel()
        return pn.Column(load_panel)

    def get_control_panel(self):
        select_buttons = pn.Row( self._select_all, self._select_region )
        clear_buttons = pn.Row( self._clear_all, self._clear_region)
        buttonbox = pn.WidgetBox( "### Selection Controls", select_buttons, clear_buttons )
        selection_mode = pn.WidgetBox("### Click-select Mode", self.blockSelection.click_select_mode )
        selection_panel = pn.Column( buttonbox, selection_mode )
        cache_panel = self.blockSelection.get_cache_panel(self.selection_mode)
        block_panels = pn.Tabs( ("select",selection_panel), ("cache",cache_panel) )
        return pn.Tabs( ("block mask", block_panels), ( "cluster mask", clm().gui()), ( "learning", clm().get_learning_panel("points")) )

    def get_block_selection_gui(self):
        return self.blockSelection.get_cache_panel(self.selection_mode)

    def get_block_selection(self) -> Optional[Dict]:
        return self.blockSelection.get_block_selecction()

    def get_cluster_panel(self):
        return clm().panel()

    def get_tile_selection_gui(self):
        basemap = spm().get_image_basemap( self.blockSelection.region_bounds )
        self.rect_grid = self.blockSelection.grid_widget()
        image = basemap * self.rect_grid * self.selected_rec
        return image

    def gui( self, **kwargs ):
        if "mode" in kwargs: self.selection_mode
        selection_mode = kwargs.get( "mode", self.selection_mode )
        self.rect0 = tm().block_index
        basemap = spm().get_image_basemap( self.blockSelection.region_bounds )
        self.rect_grid = self.blockSelection.grid_widget()
        image = basemap * self.rect_grid * self.selected_rec
        if selection_mode == BlockSelectMode.LoadTile:
            return image
        else:
            selection_panel = self.get_control_panel()
            if selection_mode == BlockSelectMode.SelectTile:
                return pn.Row( image * self.region_selection, selection_panel )
            elif selection_mode == BlockSelectMode.CreateMask:
                cluster_panel = self.get_cluster_panel()
                viz_panels = pn.Tabs( ("select", image * self.region_selection), ("cluster", cluster_panel))
                return pn.Row( viz_panels, selection_panel )
            elif selection_mode == BlockSelectMode.LoadMask:
                return self.get_block_selection_gui()

    @property
    def image_index(self) -> int:
        from spectraclass.data.base import DataManager, dm
        return dm().modal.file_selector.index

    @property
    def image_name(self) -> str:
        from spectraclass.data.base import DataManager, dm
        return dm().modal.get_image_name( self.image_index )

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

