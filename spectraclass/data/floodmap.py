
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import xarray as xa
import numpy as np
from datetime import datetime

class FloodmapProcessor:
    calendar = 'standard'
    units = 'days since 1970-01-01 00:00'

    def __init__( self, results_dir: str ):
        self.results_dir = results_dir
        self._datasets = None

    def results_file( self, fmversion: str):
        result_name = f"floodmap_comparison_{fmversion}"
        return f"{self.results_dir}/{result_name}.nc"

    @classmethod
    def pct_diff( cls,  x0: float, x1: float ) -> float:
        return (abs( x1-x0 ) * 100) / min(x0,x1)

    @classmethod
    def get_timestamp( cls, tstr: str, fmversion: str ) -> datetime:
        if fmversion == "nrt": (m, d, y) = tstr.split("-")
        elif fmversion == "legacy": (y, m, d) = tstr.split("-")
        else: raise Exception( f"Unrecognized fmversion: {fmversion}")
        return datetime(int(y), int(m), int(d))

    def filter_outliers( self, data: xa.DataArray, outliers: Optional[List[int]] = None ) -> xa.DataArray:
        return data if (outliers is None) else data.where( np.logical_not(data.lake.isin(outliers)), drop = True  )

    def get_mean(self, data: xa.DataArray, outliers: Optional[List[int]] = None ) -> xa.DataArray:
        fdata = self.filter_outliers( data, outliers )
        return fdata.mean(skipna=True).values.tolist()

    def get_lake_version_means(self, fmversion: str, varname: str ) -> xa.DataArray:
        dsets = self.get_datasets()
        return dsets[fmversion].data_vars[varname].mean(axis=0, skipna=True)

    def get_lake_means(self, varname: str ) -> Dict[str,xa.DataArray]:
        return { fmversion: self.get_lake_version_means( fmversion, varname ) for fmversion in ["legacy", 'nrt'] }

    def get_datasets(self)-> Dict[str,xa.Dataset]:
        if self._datasets is None:
            self._datasets = { fmversion: xa.open_dataset( self.results_file(fmversion) ) for fmversion in ["legacy", 'nrt'] }
        return self._datasets

    def get_vars(self, name: str, outliers: Optional[List[int]] = None )-> Dict[str,xa.DataArray]:
        dsets: Dict[str, xa.Dataset] = self.get_datasets()
        return  { fmversion: self.filter_outliers( dsets[fmversion].data_vars[ name ], outliers ) for fmversion in [ "legacy", 'nrt' ] }

    def get_means(self, outliers: Optional[List[int]] = None ):
        water_area_means = {}
        interp_area_means = {}
        pct_interp_means = {}
        dsets = self.get_datasets()
        for fmversion in [ "legacy", 'nrt' ]:
            water_area: xa.DataArray = dsets[fmversion].data_vars['water_area']
            pct_interp_array: xa.DataArray = dsets[fmversion].data_vars['pct_interp']

            water_area_means[fmversion] = self.get_mean( water_area, outliers )
            pct_interp_means[fmversion] = self.get_mean( pct_interp_array, outliers )
            interp_area_means[fmversion] = self.get_mean( pct_interp_array * water_area, outliers )

        print(f"\nMeans: {water_area_means}")
        print(f"Pct DIFF: {self.pct_diff(*list(water_area_means.values())):.2f} %")
        print(f"\nPct Interp: {pct_interp_means}")
        print(f"Pct DIFF: {self.pct_diff(*list(pct_interp_means.values())):.2f} %")
        print(f"\nInterp Area: {interp_area_means}")
        print(f"Pct DIFF: {self.pct_diff(*list(interp_area_means.values())):.2f} %")
        return dict( water_area=water_area_means, interp_area=interp_area_means, pct_interp=pct_interp_means )

    def get_interp_diff( self, outliers: Optional[List[int]] = None ):
        water_vars: Dict[str, xa.DataArray] = self.get_vars('water_area', outliers )
        interp_vars: Dict[str, xa.DataArray] = self.get_vars('pct_interp', outliers )
        lake_interp_means = {}
        water_area_means = {}
        for fmversion in ["legacy", 'nrt']:
            water_var: xa.DataArray = water_vars[fmversion]
            interp_var: xa.DataArray = interp_vars[fmversion]
            interp_area: xa.DataArray = (interp_var * water_var) / 1600
            lake_interp_means[fmversion] = interp_area.mean(axis=0, skipna=True)
            water_area_means[fmversion] = water_var.mean(axis=0, skipna=True)
        interp_diff =  lake_interp_means['nrt'] - lake_interp_means["legacy"]
        interp_diff.name = "Interpolation_Area_Difference"
        return interp_diff.dropna(dim=interp_diff.dims[0]), water_area_means
