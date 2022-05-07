import cartopy.crs as ccrs
from typing import List, Union, Tuple, Dict

def get_ccrs( p4p: Dict ) -> ccrs.CRS:
    crsid = p4p['proj']
    if crsid == 'aea':
        crs = ccrs.AlbersEqualArea( central_longitude=float(p4p['lon_0']), central_latitude=float(p4p['lat_0']),
                                    false_easting=float(p4p['x_0']), false_northing=float(p4p['y_0']),
                                    standard_parallels=( float(p4p['lat_1']), float(p4p['lat_2']) ) )
        return crs
    else:
        raise NotImplementedError( f"The crs '{crsid}' has not yet been implemented.")