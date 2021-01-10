from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass
from spectraclass.gui.spatial.google import GooglePlotManager, gpm
import xarray as xa

dm: DataManager = DataManager.initialize("demo2",'desis')
app = Spectraclass.instance()
project_data: xa.Dataset = dm.loadCurrentProject("main")
print(f"Loaded project data: vars = {project_data.variables.keys()}")

dm.save_config()
gpm().setBlock( )


