import time
import xarray as xa
import pandas as pd
from spectraclass.gui.unstructured.application import Spectraclass
from spectraclass.data.base import DataManager

app = Spectraclass.instance()
app.configure("spectraclass")

t0 = time.time()
nrows = 10

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("test")
table_cols = project_dataset.attrs['colnames']

graph_data: xa.DataArray = project_dataset["reduction"]

df = pd.DataFrame({ icol : graph_data[:,icol] for icol in range(graph_data.shape[1]) } )

print( df.head(2) )

# dropped_vars = [ vname for vname in project_data.data_vars if vname not in table_cols ]
# table_data = { tcol: project_data[tcol].values[:nrows] for tcol in table_cols }
#
# df: pd.DataFrame = pd.DataFrame( table_data, dtype='U', index=pd.Int64Index( range(nrows-1,-1,-1), name="Index" ) )
# df.insert( len(table_cols), "Class", 0, True )
# print( f"Created dataFrame  in {time.time()-t0} sec.")
#
# df.loc[ [5,7], 'Class'] = 10


# print( df )






