import cudf
from cuml.neighbors import NearestNeighbors
import xarray as xa
import numpy as np
from spectraclass.gui.application import Spectraclass
from spectraclass.data.manager import DataManager

app = Spectraclass.instance()
app.configure("spectraclass")

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
X: np.ndarray = project_dataset["reduction"].values
print(f"Input shape = {X.shape}")
X_cudf = cudf.DataFrame(X)
print(f"X_cudf head = {X_cudf.head(10)}")

# fit model
model = NearestNeighbors(n_neighbors=3)
model.fit(X)

# get 3 nearest neighbors
distances, indices = model.kneighbors(X_cudf)
print( indices.__class__ )

# print results
print(indices.head(10))
print(distances.head(10))

