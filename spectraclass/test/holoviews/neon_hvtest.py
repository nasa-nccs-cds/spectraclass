import xarray as xa
import holoviews as hv
import panel as pn
from spectraclass.gui.spatial.viewer import RasterCollectionsViewer
from spectraclass.gui.spatial.map import MapManager, mm
from spectraclass.data.spatial.tile.manager import tm
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.model.labels import LabelsManager, lm
from typing import List, Union, Tuple, Optional, Dict, Callable
pn.extension()
hv.extension('bokeh')

dm: DataManager = DataManager.initialize( "AGB", 'neon' )

block_size = 250
model_dims = 12
nepochs = 100
year= 2015
version = "beta_pmm"
roi = "541567.6_4136443.0_542567.6_4137443.0"

dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/MLBS_{year}_{roi}/MLBS_{year}_Reflectance_reflectance_warp.tif"
TileManager.block_size = block_size
TileManager.reprocess = True
dm.modal.refresh_model = True
dm.modal.model_dims = model_dims
dm.modal.modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}"
dm.modal.reduce_nepoch = nepochs

dm.loadCurrentProject()
classes = [ ('air', "cyan"),
            ('water', "blue"),
            ('fire', "magenta"),
            ('earth', "green") ]

lm().setLabels( classes )

dset_names: List[str] = list(dm.modal.datasets.keys())
fdata: xa.DataArray = mm().getModelData(raster=True,norm=True).rename( dict(band='feature') )
band_data: xa.DataArray =  mm().getBandData(raster=True,norm=True)
reproduction=mm().getReproduction(raster=True)
viewer = RasterCollectionsViewer( dict( features=fdata, bands=band_data, reproduction=reproduction ), verification=reproduction )
viewer.panel()