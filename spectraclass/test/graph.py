import time
import xarray as xa
import numpy as np
from spectraclass.gui.application import Spectraclass
from spectraclass.data.manager import DataManager
from spectraclass.graph.base import ActivationFlow

app = Spectraclass.instance()
app.configure("spectraclass")
n_neighbors = 5
t0 = time.time()
source = np.array( [30], dtype = np.int32 )

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
# table_cols = project_dataset.attrs['colnames']

graph_data: xa.DataArray = project_dataset["reduction"]
activation_flow = ActivationFlow.instance( graph_data, n_neighbors )
activation_flow.spread( source )
print( "Completed" )


