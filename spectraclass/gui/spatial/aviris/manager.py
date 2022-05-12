from spectraclass.data.base import DataManager
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from spectraclass.gui.control import UserFeedbackManager, ufm
import numpy as np
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from cartopy.mpl.geoaxes import GeoAxes
from spectraclass.xext.xgeo import XGeo
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.spatial.tile.manager import TileManager, tm
import matplotlib.pyplot as plt
import ipywidgets as ipw
import time, xarray as xa

class AvirisDatasetManager:

    @log_timing
    def __init__(self, **kwargs):
        self.dm: DataManager = DataManager.initialize( "DatasetManager", 'aviris' )
        if "cache_dir" in kwargs: self.dm.modal.cache_dir = kwargs["cache_dir"]
        if "data_dir"  in kwargs: self.dm.modal.data_dir = kwargs["data_dir"]
        if "images_glob" in kwargs: self.dm.modal.images_glob = kwargs["images_glob"]
        self.dm.modal.ext = kwargs.get( "ext", "_img" )
        self.init_band = kwargs.get( "init_band", 160 )
        self.dm.proc_type = "cpu"
        TileManager.block_size = kwargs.get( 'block_size',  250 )
        self.nimages = len( self.dm.modal.image_names )
        self._nbands = None
        self.band_selector: ipw.IntSlider = None
        self.band_plot: AxesImage = None
        self._axes: Axes = None
        lgm().log( f"AvirisDatasetManager: Found {self.nimages} images "  )

    @property
    def nbands(self) -> int:
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

    def update_image( self ):
        self.dm.modal.set_current_image( self.image_index )
        data_array: xa.DataArray = tm().tile.data
        band_array: np.ndarray = data_array[self.band_index].values.squeeze()
        nodata = data_array.attrs.get('_FillValue')
        band_array[band_array == nodata] = np.nan
        if self.band_plot is None:  self.band_plot = self.ax.imshow( band_array, cmap="jet")
        else:                       self.band_plot.set_data( band_array )

    def on_image_change( self, event: Dict ):
        ufm().show( f"Loading image {self.image_name}" )
        self.update_image()

    def on_band_change( self, event: Dict ):
        ufm().show( f"Plotting band {self.band_index}" )
        self.update_image()

    def gui(self):
        self.band_selector = ipw.IntSlider( self.init_band, 0, self.nbands, 1 )
        self.band_selector.observe( self.on_band_change, "value" )
        self.dm.modal.set_file_selection_observer( self.on_image_change )
        control_panel = ipw.VBox( [ufm().gui(), self.dm.modal.file_selector, self.band_selector] )
        widget = ipw.HBox( [self.band_plot.figure.canvas, control_panel], layout=ipw.Layout(flex='1 1 auto') )
        plt.ion()
        return widget

    @property
    def ax(self) -> Axes:
        if self._axes is None:
            plt.ioff()
            self._axes: Axes = plt.axes()
            self._axes.set_yticks([])
            self._axes.set_xticks([])
        return self._axes