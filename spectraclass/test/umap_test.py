import time
import xarray as xa
from spectraclass.gui.application import Spectraclass
from spectraclass.data.manager import DataManager
from spectraclass.reduction.base import UMAP

app = Spectraclass.instance()
app.configure("spectraclass")
n_neighbors = 10
project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
umap_data: xa.DataArray = project_dataset["reduction"].compute()

umap = UMAP.instance( n_neighbors=n_neighbors,  n_components=3 )
embedding = umap.embed( umap_data )
print( embedding.__class__.__name__ )





