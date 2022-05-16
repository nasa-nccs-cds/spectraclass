import numpy as np, os
from osgeo import osr
import xarray as xr
from typing import Dict, List, Tuple, Union, Optional

class XExtension(object):
    """  This is the base class for xarray extensions """

    StandardAxisNames = { 'x': [ 'x', 'xc', 'lon' ], 'y': [ 'y', 'yc', 'lat' ], 't': [ 't', 'time' ] }

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj: xr.DataArray = xarray_obj
        self.y_coord = self.getCoordName('y')
        self.x_coord = self.getCoordName('x')
        self.time_coord = self.getCoordName('t')
        self._crs: osr.SpatialReference = self.getSpatialReference()
        self._geotransform = self.getTransform()

    def set_persistent_attribute(self, name: str, value: str ):
        self._obj.attrs[ name ] = value

    def get_persistent_attribute(self, name: str ) -> Optional[str]:
        return self._obj.attrs.get( name )

    @property
    def y_inverted(self)-> Optional[bool]:
        if self.y_coord is None: return None
        return self.ycoords[0] > self.ycoords[-1]

    @property
    def xcoords(self)-> Optional[np.ndarray]:
        if self.x_coord is None: return None
        return self._obj[self.x_coord].values

    @property
    def ycoords(self)-> Optional[np.ndarray]:
        if self.y_coord is None: return None
        return self._obj[self.y_coord].values

    def getCoordName( self, axis: str ) -> Optional[str]:
        for cname, coord in self._obj.coords.items():
            if (str(cname).lower() in self.StandardAxisNames[axis]) or (coord.attrs.get("axis") == axis) or (axis == cname):
                return str(cname)
        return None

    def getSpatialReference( self ) -> osr.SpatialReference:
        sref = osr.SpatialReference()
        crs = self._obj.attrs.get('crs')
        if crs is None:
            if hasattr( self._obj, 'spatial_ref'):
                sr = self._obj.spatial_ref
                crs_wkt = sr.attrs.get( "crs_wkt", sr.attrs.get( "spatial_ref", None ) )
                if crs_wkt: sref.ImportFromWkt( crs_wkt )
                else: sref.ImportFromEPSG(4326)
            else:
                sref.ImportFromEPSG(4326)
        else:
            if "epsg" in crs.lower():
                espg = int(crs.split(":")[-1])
                sref.ImportFromEPSG(espg)
            elif "+proj" in crs.lower():
                sref.ImportFromProj4(crs)
            else:
                raise Exception(f"Unrecognized crs: {crs}")
        return sref

    @property
    def resolution(self):
        transform = self.getTransform()
        return [ transform[1], -transform[5] ]

    def getTransform(self):
        transform = self._obj.attrs.get('transform')
        if transform is None:
            y_arr = self._obj.coords[ self.y_coord ]
            x_arr = self._obj.coords[ self.x_coord ]
            res = self._obj.attrs.get('res')

            if y_arr.ndim < 2:
                x_2d, y_2d = np.meshgrid(x_arr, y_arr)
            else:
                x_2d = x_arr
                y_2d = y_arr

            x_cell_size = np.nanmean(np.absolute(np.diff(x_2d, axis=1))) if res is None else res[1]
            y_cell_size = np.nanmean(np.absolute(np.diff(y_2d, axis=0))) if res is None else res[0]

            min_x_tl = x_2d[0, 0] - x_cell_size / 2.0
            max_y_tl = y_2d[0, 0] + y_cell_size / 2.0
            transform = min_x_tl, x_cell_size, 0, max_y_tl, 0, -y_cell_size
            self._obj.attrs['transform'] = transform
        return transform
