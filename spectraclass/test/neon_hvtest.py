import xarray as xa
import holoviews as hv
import panel as pn
from spectraclass.gui.spatial.viewer import RasterCollectionsViewer
from spectraclass.gui.spatial.map import MapManager, mm
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager
from spectraclass.model.labels import LabelsManager, lm
from typing import List, Union, Tuple, Optional, Dict, Callable
pn.extension()
hv.extension('bokeh')

dm: DataManager = DataManager.initialize( "AGB", 'neon' )

block_size = 250
model_dims = 6
year= 2015
version = "beta_pmm"
roi = "541567.6_4136443.0_542567.6_4137443.0"

dm.proc_type = "cpu"
dm.modal.images_glob = f"AGB/test/{version}/MLBS_{year}_{roi}/MLBS_{year}_Reflectance_reflectance_warp.tif"
TileManager.block_size = block_size
TileManager.reprocess = False
dm.modal.refresh_model = False
dm.modal.model_dims = model_dims
dm.modal.modelkey = f"agp.neon.{version}.{year}.{roi}.{block_size}"

dm.loadCurrentProject()
classes = [ ('air', "cyan"),
            ('water', "blue"),
            ('fire', "magenta"),
            ('earth', "green") ]

lm().setLabels( classes )

dset_names: List[str] = list(dm.modal.datasets.keys())
dset: Dict[str,xa.DataArray] = dm.modal.datasets[ dset_names[0] ]
fdata: xa.DataArray = mm().data.rename( dict(band='feature') )
sdata: xa.DataArray = dset['raw']
viewer = RasterCollectionsViewer( dict( features=fdata, bands=sdata, reproduction=mm().getReproduction(True) ) )
viewer.panel()