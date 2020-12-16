import cudf
from cuml.neighbors import NearestNeighbors
import xarray as xa
import numpy as np
from spectraclass.gui.application import Spectraclass
from spectraclass.data.base import DataManager

app = Spectraclass.instance()
app.configure("spectraclass")
nVerts = 10000

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
full_input = project_dataset["reduction"]
X = full_input.values[0:nVerts,:]
print(f"Dataset shape = {full_input.shape}, input shape = {X.shape}, ")
X_cudf = cudf.DataFrame(X)
print(f"X_cudf head = {X_cudf.head(10)}")
app.show_gpu_usage()

# fit model
model = NearestNeighbors(n_neighbors=3)
model.fit(X)
app.show_gpu_usage()

# get 3 nearest neighbors
distances, indices = model.kneighbors(X_cudf)
app.show_gpu_usage()
print( indices.__class__ )

# print results
print(indices.head(10))
print(distances.head(10))

