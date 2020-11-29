import pandas as pd
import numpy as np
import xarray as xa
from spectraclass.gui.application import Spectraclass
from spectraclass.data.manager import DataManager

app = Spectraclass.instance()
app.configure("spectraclass")
nrows = 20000

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
input_data: xa.DataArray = project_dataset["reduction"]
input_data[0:nrows,:].to_dataframe("spectral_embedding").to_csv("/tmp/spectraclass_data.csv")

