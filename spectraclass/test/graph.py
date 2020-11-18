import time
import xarray as xa
from spectraclass.gui.application import Astrolab
from spectraclass.data.manager import DataManager
from spectraclass.graph.base import ActivationFlow

app = Astrolab.instance()
app.configure("spectraclass")
n_neighbors = 5
t0 = time.time()

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
# table_cols = project_dataset.attrs['colnames']

graph_data: xa.DataArray = project_dataset["reduction"]
activation_flow = ActivationFlow.instance( graph_data, n_neighbors )


