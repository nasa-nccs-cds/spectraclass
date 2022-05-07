#%%

from bokeh.plotting import figure, output_file, show
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import xarray as xa
import numpy as np
import hvplot.xarray
from datetime import datetime
import pandas as pd
from spectraclass.data.floodmap import FloodmapProcessor

results_dir = "/Users/tpmaxwel/Development/Data/WaterMapping/Results"
fmp = FloodmapProcessor( results_dir )
pct_interp_map: Dict[str,xa.DataArray] = fmp.get_lake_means( 'pct_interp' )
lake_specs = f"{results_dir}/lake_locations.csv"
df = pd.read_csv( lake_specs )

fmversion = 'nrt'
smax = df['size'].max()
pct_interp_array: xa.DataArray = pct_interp_map[fmversion]

lines = []
alphas = []
for iLake in range(100):
    lake_index = pct_interp_array.lake.values[iLake]
    row = df.loc[df['index'] == lake_index]
    size, lat = row['size'].iloc[0], row['lat'].iloc[0]
    pct_interp = pct_interp_array.values[ iLake ]
    alpha = min( (size*35)/smax, 1.0 )
    lspec = ( [ [lat,lat], [0,pct_interp] ], alpha )
    lines.append( lspec )
    print( f"{iLake}[{lake_index}]: {lspec}")