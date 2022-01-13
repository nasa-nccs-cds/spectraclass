import xarray as xa
import numpy as np

data_dir = "/Users/tpmaxwel/Development/Cache/spectraclass/DESIS"
bk = "500-500-0-0-m32"
def dfile(idx): return f"{data_dir}/DESIS-HSI-L1C-DT0468853252_00{idx}-20200628T153803-V0210_b-{bk}.nc"

for idx in [2,3]:
    data_file = dfile(idx)
    dset: xa.Dataset = xa.open_dataset( data_file )
    raw: xa.DataArray = dset["raw"]
    print(f"Reading file {data_file}, raw shape = {raw.shape}, dims = {raw.dims}")
    data_array: np.ndarray = raw.values
    print( ( np.nanmax( data_array ), np.nanmin( data_array ), np.nansum( data_array ), np.nanstd( data_array ) ) )