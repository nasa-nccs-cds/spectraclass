import cudf, os
from cuml.neighbors import NearestNeighbors
import xarray as xa
from spectraclass.gui.unstructured.application import Spectraclass
from spectraclass.data.base import DataManager

app = Spectraclass.instance()
app.configure("spectraclass")
nneighbors = 5
nverts = 10000

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
X = project_dataset["reduction"].values[0:nverts,:]
print(f"Dataset input shape = {X.shape}, ")

X_cudf = cudf.DataFrame(X)
print( f"\nINPUT DATA:\n{X_cudf.head(10)}")

# fit model
model = NearestNeighbors(n_neighbors=nneighbors)
model.fit(X)

# get 3 nearest neighbors
distances, indices = model.kneighbors(X_cudf)
os.system("nvidia-smi")

# print results
print( f"\nINDICES:\n{indices.head(10)}")
print(f"\nDISTANCES:\n{distances.head(10)}")
