import time
import xarray as xa
from spectraclass.gui.application import Spectraclass
from spectraclass.data.manager import DataManager
from spectraclass.reduction.base import UMAP
import cudf, cuml, cupy, cupyx

app = Spectraclass.instance()
app.configure("spectraclass")
n_neighbors = 10
project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
umap_data: xa.DataArray = project_dataset["reduction"].compute()

umap = UMAP.instance()
embedding = umap.transform( umap_data )
print( f"Completed embedding, result shape = {embedding.shape}" )




