import xarray as xa
from spectraclass.gui.unstructured.application import Spectraclass
from spectraclass.data.base import DataManager
from spectraclass.reduction.base import UMAP

app = Spectraclass.instance()
app.configure("spectraclass")
n_neighbors = 10
project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
umap_data: xa.DataArray = project_dataset["reduction"].compute()

umap = UMAP.instance( n_neighbors=n_neighbors,  n_components=3 )
umap.embed( umap_data )
print( "Embedding[0:10]:" )
print( umap.embedding[0:10] )





