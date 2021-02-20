import os, time, random, numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type
import xarray as xa
TEST_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def load_test_data( dataset_type: str, dsid: str, data_type: str, ilayer: int ) -> xa.DataArray:
    t0 = time.time()
    data_file = os.path.join(TEST_DIR, "data", dataset_type, f"{dsid}_{data_type}_{ilayer}.nc4")
    dataset = xa.load_dataset(data_file)
    print(f" loaded test data in time {time.time() - t0} sec")
    return dataset['data']
