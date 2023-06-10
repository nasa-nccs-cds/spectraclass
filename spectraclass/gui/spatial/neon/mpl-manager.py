from spectraclass.gui.control import UserFeedbackManager, ufm
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from cartopy.mpl.geoaxes import GeoAxes
from spectraclass.xext.xgeo import XGeo
from typing import List, Union, Tuple, Optional, Dict, Callable
import panel as pn, holoviews as hv
import time, xarray as xa

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


    @property
    def image_index(self) -> int:
        from spectraclass.data.base import DataManager, dm
        return dm().modal.file_selector.index

    @property
    def image_name(self) -> str:
        from spectraclass.data.base import DataManager, dm
        return dm().modal.get_image_name( self.image_index )


    def update_image( self ):
        from spectraclass.gui.spatial.map import MapManager, mm
        band_array = self.get_band_data()
        vmean, vstd = np.nanmean(band_array), np.nanstd( band_array )
        vrange = [ max(vmean-2*vstd, 0.0), vmean+2*vstd ]
        (nr,nc) = band_array.shape
        extent = ( -0.5, nc-0.5, -0.5, nr-0.5 )
        if self.band_plot is None:
            self.band_plot = self.ax.imshow( band_array, cmap="jet", origin="lower", extent=extent)
        else:
            self.band_plot.set_extent( extent )
            self.band_plot.set_data( band_array )
            self.ax.set_xbound(extent[0],extent[1])
            self.ax.set_ybound(extent[2],extent[3])
        self.band_plot.set_clim(*vrange)
        self.add_block_selection()
        mm().image_update()
      #  self.select_block()

    @log_timing
    def on_image_change( self, event: Dict ):
        ufm().show( f" ** Loading image {self.image_name}" )
        self.clear_block_cache()
        self.update_image()

    def clear_block_cache(self):
        self._transformed_block_data = None

    def gui(self):
        self.update_image()
        return self.band_plot.figure.canvas





    def overlay_image_data(self) -> xa.DataArray:
        if self._transformed_block_data is None:
            block_data = self.get_data_block()
            self._transformed_block_data: xa.DataArray = block_data.xgeo.reproject(espg=3785)
        result =  self._transformed_block_data[self.band_index].squeeze()
        nnan = np.count_nonzero(np.isnan(result.values))
        lgm().log(f"EXT: overlay_image_data, %nan: {(nnan*100.0)/result.size}")
        return result

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

    def on_pick(self, event: PickEvent =None):
        lgm().log( f" Pick Event: type = {type(event)}" )
        if type(event.artist) == Rectangle:
            self.select_block( event.artist )

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
        self._blocks: Dict[Tuple[int,int],Rectangle] = {}
        self._transformed_block_data = None
        self._selected_block: Tuple[int,int] = (0,0)
        TileManager.block_size = kwargs.get( 'block_size',  250 )
        self.nimages = len( self.dm.modal.image_names )
        self._nbands = None
        lgm().log( f"AvirisDatasetManager: Found {self.nimages} images "  )

    @property
    def selected_block(self) -> Optional[Rectangle]:
        return self._blocks.get( self._selected_block )

    def get_block(self, coords: Tuple[int,int] ) -> Optional[Rectangle]:
        return self._blocks.get( coords )

    @property
    def nbands(self) -> int:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        if self._nbands is None:
            data_array: xa.DataArray = tm().tile.data
            self._nbands = data_array.shape[0]
        return self._nbands

    @property
    def band_index(self) -> int:
        if self.band_selector is None:
            return self.init_band
        else:
            return self.band_selector.value

    @property
    def image_index(self) -> int:
        return self.dm.modal.file_selector.index

    @property
    def image_name(self) -> str:
        return self.dm.modal.get_image_name( self.image_index )

    def get_band_data(self) -> np.ndarray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        data_array: xa.DataArray = tm().tile.data
        band_array: np.ndarray = data_array[self.band_index].values.squeeze().transpose()
        nodata = data_array.attrs.get('_FillValue')
        band_array[band_array == nodata] = np.nan
        return band_array

    def update_image( self ):
        from spectraclass.gui.spatial.map import MapManager, mm
        band_array = self.get_band_data()
        self.dm.modal.set_current_image( self.image_index )
        vmean, vstd = np.nanmean(band_array), np.nanstd( band_array )
        vrange = [ max(vmean-2*vstd, 0.0), vmean+2*vstd ]
        if self.band_plot is None:  self.band_plot = self.ax.imshow( band_array, zorder=2.0, cmap="jet")
        else:                       self.band_plot.set_data( band_array )
        self.band_plot.set_clim(*vrange)
        mm().image_update()
        self.select_block( new_image = True )

    @log_timing
    def on_image_change( self, event: Dict ):
        ufm().show( f" *** Loading image {self.image_name}" )
        self.clear_block_cache()
        self.update_image()

    def clear_block_cache(self):
        self._transformed_block_data = None

    @log_timing
    def on_band_change( self, event: Dict ):
        ufm().show( f"Plotting band {self.band_index}" )
        self.update_overlay_band()

    @log_timing
    def on_transparency_change( self, event: Dict ):
        self.overlay_plot.set_alpha( event['new'] )
        self.overlay_plot.figure.canvas.draw_idle()

    def gui(self):
        plt.ioff()
        self.update_image()
        self.add_block_selection()
        self.select_block()
        self.band_selector = ipw.IntSlider( self.init_band, min=0, max=self.nbands, step=1 )
        self.band_selector.observe( self.on_band_change, "value" )
        self.alpha_selector = ipw.FloatSlider( 1.0, min=0.0, max=1.0, step=0.1 )
        self.alpha_selector.observe(self.on_transparency_change, "value")
        self.dm.modal.set_file_selection_observer( self.on_image_change )
        control_panel = ipw.VBox( [ufm().gui(), self.dm.modal.file_selector, self.band_plot.figure.canvas] )
        image_controls = ipw.VBox( [ ipw.HBox( [ ipw.Label("band"), self.band_selector ] ), ipw.HBox( [ ipw.Label("alpha"), self.alpha_selector ] ) ]  )
        overlay_panel = ipw.VBox( [ self.overlay_plot.figure.canvas, image_controls ] )
        widget = ipw.HBox( [ overlay_panel, control_panel ], layout=ipw.Layout(flex='1 1 auto') )
        plt.ion()
        return widget

    @property
    def ax(self) -> Axes:
        if self._axes is None:
            self._axes: Axes = plt.axes()
            self._axes.set_yticks([])
            self._axes.set_xticks([])
            self._axes.figure.canvas.mpl_connect( 'pick_event', self.on_pick )
        return self._axes

    @exception_handled
    def add_block_selection(self):
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        transform = tm().tile.data.attrs['transform']
        block_size = tm().block_size
        block_dims = tm().block_dims
        lgm().log(f"  add_block_selection: block_size={block_size}, block_dims={block_dims}, transform={transform} ")
        for tx in range( block_dims[0] ):
            for ty in range( block_dims[1] ):
                selected = ( (tx,ty) == self._selected_block )
                ix, iy = tx*block_size, ty*block_size
                lw = self.slw if ( selected ) else 1
                color = self.selection_color if ( selected ) else self.grid_color
                r = Rectangle( (iy, ix), block_size, block_size, fill=False, edgecolor=color, lw=lw, alpha=1.0 )
                setattr( r, 'block_index', (tx,ty) )
                r.set_picker( True )
                self.ax.add_patch( r )
                self._blocks[(tx,ty)] = r

    def highlight_block( self, r: Rectangle ):
        srec = self.selected_block
        if srec is not None:
            srec.set_color( self.grid_color )
        r.set_linewidth(self.slw)
        r.set_color( self.selection_color )
        self._selected_block = r.block_index
        self.ax.figure.canvas.draw_idle()

    @log_timing
    def select_block(self, r: Rectangle = None ):
        from spectraclass.data.spatial.manager import SpatialDataManager
        if r is not None:
            self._selected_block = r.block_index
            self.clear_block_cache()
            self.highlight_block( r )
        band_data = self.overlay_image_data()
        ext = SpatialDataManager.extent( band_data )
        norm = Normalize(**self.get_color_bounds(band_data))
        if self.overlay_plot is None:
            lgm().log(f"EXT: init_block--> Set bounds: {(ext[0], ext[1])}  {(ext[2], ext[3])}")
            self.base.setup_plot( "Subtile overlay", ( ext[0], ext[1] ), ( ext[2], ext[3] ), slider=False )
            self.overlay_plot = band_data.plot.imshow(ax=self.base.gax, alpha=1.0, zorder=2.0, cmap='jet', norm=norm, add_colorbar=False)
        else:
            ufm().show(f"!Loading block {self._selected_block}" )
            lgm().log( f"EXT: select_block[{self._selected_block}]--> Set bounds: {( ext[0], ext[1] )}  {( ext[2], ext[3] )}")
            self.base.gax.set_xbound( ext[0], ext[1] )
            self.base.gax.set_ybound( ext[2], ext[3] )
            self.base.basemap.set_extent( ext )
            lgm().log(f"EXT: overlay_plot--> set band data: shape={band_data.shape}")
            self.overlay_plot.set_extent( ext )
            self.overlay_plot.set_data(band_data.values)
            self.overlay_plot.figure.canvas.draw_idle()

    def overlay_image_data(self) -> xa.DataArray:
        if self._transformed_block_data is None:
            block_data = self.get_data_block()
            self._transformed_block_data: xa.DataArray = block_data.xgeo.reproject(espg=3785)
        result =  self._transformed_block_data[self.band_index].squeeze()
        nnan = np.count_nonzero(np.isnan(result.values))
        lgm().log(f"EXT: overlay_image_data, %nan: {(nnan*100.0)/result.size}")
        return result

    def update_overlay_band(self):
        band_data = self.overlay_image_data()
        self.overlay_plot.set_data( band_data.values )
        self.overlay_plot.figure.canvas.draw_idle()

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

    def on_pick(self, event: PickEvent =None):
        lgm().log( f" Pick Event: type = {type(event)}" )
        if type(event.artist) == Rectangle:
            self.select_block( event.artist )
