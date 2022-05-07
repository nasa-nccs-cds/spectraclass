from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass
import xarray as xa

dm: DataManager = DataManager.initialize("demo2",'desis')
app = Spectraclass.instance()
project_data: xa.Dataset = dm.loadCurrentProject("main")
print(f"Loaded project data: vars = {project_data.variables.keys()}")
dm.save_config()



