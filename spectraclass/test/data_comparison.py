import cudf
from cuml.datasets import make_blobs
import pandas as pd
import numpy as np
import xarray as xa
from spectraclass.gui.application import Spectraclass
from spectraclass.data.base import DataManager

app = Spectraclass.instance()
app.configure("spectraclass")
random_state = 0

project_dataset: xa.Dataset = DataManager.instance().loadCurrentProject("spectraclass")
spectraclass_data = project_dataset["reduction"].values
print(f"Spectraclass input shape = {spectraclass_data.shape}")
sc_cudf_data = cudf.DataFrame(spectraclass_data)
scindx = sc_cudf_data.index
print( f"Spectraclass dataframe: ndim = {sc_cudf_data.ndim}, size = {sc_cudf_data.size}, col shape = {sc_cudf_data.columns.shape}, shape = {sc_cudf_data.index.shape}, np-shape = {sc_cudf_data.values.shape}")
print( f"  --> index: shape = {scindx.shape}, ndim = {scindx.ndim}, dtype = {scindx.dtype}, size = {scindx.size}, is_contiguous = {scindx.is_contiguous}")

n_samples = spectraclass_data.shape[0]
n_features = spectraclass_data.shape[1]
scipy_data, _ = make_blobs(n_samples=n_samples,  n_features=n_features,  centers=5, random_state=random_state)
print(f"Scipy input shape = {scipy_data.shape}")
scipy_cudf_data = cudf.DataFrame(scipy_data)
spyindx = scipy_cudf_data.index

print( f"scipy dataframe: ndim = {scipy_cudf_data.ndim}, size = {scipy_cudf_data.size}, col shape = {scipy_cudf_data.columns.shape}, shape = {scipy_cudf_data.index.shape}, np-shape = {scipy_cudf_data.values.shape}")
print( f"  --> index: shape = {spyindx.shape}, ndim = {spyindx.ndim}, dtype = {spyindx.dtype}, size = {spyindx.size}, is_contiguous = {spyindx.is_contiguous}")

print( f"\nDifference = {scindx.difference(spyindx)}")
