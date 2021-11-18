from spectraclass.data.base import DataManager
from spectraclass.application.controller import app
import xarray as xa
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.gui.spatial.map import MapManager, mm

dm: DataManager = DataManager.initialize( "demo2", 'desis' )
dm.loadCurrentProject("main")

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]

lm().setLabels( classes )
appgui = app().gui()
map_mgr = mm()

raster: xa.DataArray = map_mgr.block.data
cx, cy = raster.x.data[300], raster.y.data[300]
pid = map_mgr.block.coords2pindex(cy,cx)
ptindices = map_mgr.block.pindex2indices(pid)
classification = map_mgr.label_map.values[ ptindices['iy'], ptindices['ix'] ]