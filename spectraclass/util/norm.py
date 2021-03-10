import numpy as np
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict

def scale_to_bounds( raster: xa.DataArray, bounds: Tuple[float, float], axis: int, fill_value = None ) -> xa.DataArray:
    vmin = raster.min(dim=raster.dims[axis], skipna=True)
    vmax = raster.max(dim=raster.dims[axis], skipna=True)
    scale = (bounds[1] - bounds[0]) / (vmax - vmin)
    result: xa.DataArray = (raster - vmin) * scale + bounds[0]
    result.attrs.update( raster.attrs )
    if fill_value is not None:
        result = result.fillna( fill_value )
    return result