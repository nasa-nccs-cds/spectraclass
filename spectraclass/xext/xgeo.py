import math

import numpy as np, os
from ..util.configuration import  Region
from ..util.crs import CRS
from typing import Dict, List, Tuple
from osgeo import osr, gdalconst, gdal
from pyproj import Proj, transform, Transformer
from .grid import GDALGrid
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from shapely.geometry import Polygon
import xarray as xr
from .xextension import XExtension
from affine import Affine
import rasterio, osgeo
from rasterio.warp import reproject, Resampling, transform, calculate_default_transform, transform_bounds
sqrt2 = math.sqrt(2.0)

@xr.register_dataarray_accessor('xgeo')
class XGeo(XExtension):
    """  This is an extension for xarray to provide an interface to GDAL capabilities """

    def __init__(self, xarray_obj: xr.DataArray):
        XExtension.__init__( self, xarray_obj )

    @classmethod
    def loadRasterFile( cls, filePath: str, **args ) -> xr.DataArray:
        dirName = os.path.basename(os.path.dirname(filePath))
        name: str = args.get( "name", os.path.basename( filePath ) )
        band: int = args.get("band",-1)
        grid = GDALGrid( filePath )
        if name is None: name = os.path.basename(filePath)
        return grid.xarray( name, band )

    @classmethod
    def loadRasterFiles( cls, filePaths: List[str], **args ) -> List[xr.DataArray]:
        bbox: Region = args.get("region")
        if bbox is None:
            data_arrays: List[xr.DataArray] = [ cls.loadRasterFile( file, **args ) for file in filePaths]
        else:
            data_arrays: List[xr.DataArray] = [ cls.loadRasterFile( file, **args )[ bbox.origin[1]:bbox.bounds[1], bbox.origin[0]:bbox.bounds[0] ] for file in filePaths]
        return data_arrays

    def countInstances(self, values: List[int] ) -> xr.DataArray:
        counts = ( [( self._obj == cval ).sum(dim=[self.x_coord,self.y_coord],keep_attrs=True) for cval in values] )
        return xr.concat( counts, 'counts' ).transpose()

    def crop(self, minx: float, miny: float, maxx: float, maxy: float, buffer: float = 0 ) -> xr.DataArray:
        xbnds = [ minx - buffer, maxx + buffer ]
        ybnds = [ maxy + buffer, miny - buffer ] # if self._y_inverted else  [ miny - buffer, maxy + buffer ]
        args = { self.x_coord: slice(*xbnds), self.y_coord: slice(*ybnds)  }
        return self._obj.sel( args )

    def getUTMProj(self) -> osr.SpatialReference:
        y_arr = self._obj.coords[self.y_coord]
        x_arr = self._obj.coords[self.x_coord]
        latitude =  (y_arr[0] + y_arr[-1]) / 2.0
        longitude = (x_arr[0] + x_arr[-1]) / 2.0
        return CRS.get_utm_sref( longitude, latitude )

    def to_utm( self, resolution: Tuple[float,float], **kwargs ) -> xr.DataArray:
        utm_sref: osr.SpatialReference = kwargs.get( 'sref', self.getUTMProj() )
        gdalWaterMask: GDALGrid = self.to_gdalGrid()
        utmGdalWaterMask = gdalWaterMask.reproject( utm_sref, resolution=resolution )
        result =  utmGdalWaterMask.xarray( f"{self._obj.name}-utm", time_axis =self._obj.coords["time"] )
        result.attrs['SpatialReference'] = utm_sref
        result.attrs['resolution'] = resolution
        return result

    def gdal_reproject( self, **kwargs ) -> xr.DataArray:
        sref = osr.SpatialReference()
        proj4 = kwargs.get( 'proj4', None )
        espg =  kwargs.get( 'espg',  4326 )
        if proj4 is not None:  sref.ImportFromProj4( proj4 )
        else:                  sref.ImportFromEPSG( espg )
        gdalGrid: GDALGrid = self.to_gdalGrid()
        rGdalGrid = gdalGrid.to_projection( sref )
        result =  rGdalGrid.xarray( f"{self._obj.name}" )
        result.attrs['SpatialReference'] = sref
        return result

    def get_dest_transform(self, src_crs, dst_crs, src_shape ):
        bounds = [np.nanmin(self.xcoords), np.nanmin(self.ycoords), np.nanmax(self.xcoords), np.nanmax(self.ycoords)]
        try:
            (x0, y0, x1, y1) = transform_bounds( src_crs, dst_crs, *bounds )  # 0:left, 1:bottom, 2:right, 3:top
        except Exception as err:
            lgm().log( f"Error transforming coordinates: {err}")
            (x0, y0, x1, y1) = bounds
        dst_shape = [src_shape[0], math.ceil(src_shape[1] * sqrt2), math.ceil(src_shape[2] * sqrt2)]
        dx, dy = (x1 - x0) / (dst_shape[2] - 1), (y1 - y0) / (dst_shape[1] - 1),
        return [dx, 0.0, x0, 0.0, -dy, y1]

    def reproject( self, **kwargs ) -> xr.DataArray:
        from rasterio.crs import CRS
        sref = osr.SpatialReference()
        proj4 = kwargs.get( 'proj4', None )
        espg =  kwargs.get( 'espg',  4326 )
        if proj4 is not None:  sref.ImportFromProj4( proj4 )
        else:                  sref.ImportFromEPSG( espg )
        with rasterio.Env():
            src_shape = self._obj.shape
            src_transform = self._obj.transform[0:6]
            src_sref: osgeo.osr.SpatialReference = self._crs
            source: np.ndarray = self._obj.data
            src_crs =  CRS.from_wkt( src_sref.ExportToWkt() )
            dst_crs =  CRS.from_wkt( sref.ExportToWkt() )

            dst_transform = self.get_dest_transform( src_crs, dst_crs, src_shape )
            dst_shape =  [ src_shape[0], math.ceil(src_shape[1]*sqrt2), math.ceil(src_shape[2]*sqrt2)]
            destination = np.zeros( dst_shape, np.float32 )

            lgm().log( f" XGEO.reproject-> src_shape:{src_shape} -> dst_shape:{dst_shape}")
            lgm().log(f" -->  src crs:  {src_crs}")
            lgm().log(f" --> dest crs:  {dst_crs}")
            lgm().log(f" -->  src transform:  {src_transform}")
            lgm().log(f" --> dest transform:  {dst_transform}")

            reproject(
                source,
                destination,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                dst_nodata = np.nan)

            [ dx, _, x0, _, dy, y0 ] = dst_transform
            [ ny, nx ]= destination.shape[1:]
            x = np.linspace( x0, x0+dx*nx, nx )
            y = np.linspace( y0, y0+dy*ny, ny )
            [ bd, yd, xd ] = self._obj.dims
            band = self._obj.coords[ bd ]
            coords = {bd:band, yd:y, xd:x}
            vrange = [np.nanmin(destination),np.nanmax(destination)]
            lgm().log(f" --> result vrange:  {vrange}")

            result = xr.DataArray( destination, dims=self._obj.dims, coords =coords )
            result.attrs['transform'] = dst_transform
            result.attrs['crs'] = dst_crs
            return result

    @property
    def geographic_sref(self):
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(4326)
        return sref

    def project_to_geographic( self, x_coord: float, y_coord: float ) -> Tuple[float,float]:
        return self.project_coords( x_coord, y_coord, self.geographic_sref )

    def project_coords( self, x_coord: float, y_coord: float, sref: osr.SpatialReference ) -> Tuple[float,float]:
        trans = osr.CoordinateTransformation( self._crs, sref )
        return trans.TransformPoint( x_coord, y_coord )[:2]

    def resample_to_target(self, target: xr.DataArray, dims_map: Dict[str,str]) -> xr.DataArray:
        dim_args = { dim0: target[dim1] for dim0,dim1 in dims_map.items() }
        return self._obj.interp(**dim_args)

    def to_gdal(self) -> gdal.Dataset:
        in_array: np.ndarray = self._obj.values
        num_bands = 1
        nodata_value = self._obj.attrs.get('nodatavals',[None])[0]
        gdal_dtype = gdalconst.GDT_Float32
        proj = self._crs.ExportToWkt()

        if in_array.ndim == 3:  num_bands, y_size, x_size = in_array.shape
        else:                   y_size, x_size = in_array.shape

        dataset: gdal.Dataset = gdal.GetDriverByName('MEM').Create("GdalDataset", x_size, y_size, num_bands, gdal_dtype)

        lgm().log( f"  ** XGEO.to_gdal: transform = {self._geotransform}")
        dataset.SetGeoTransform( self._geotransform[:6] )
        dataset.SetProjection( proj )

        if in_array.ndim == 3:
            for band in range(1, num_bands + 1):
                rband = dataset.GetRasterBand(band)
                rband.WriteArray(in_array[band - 1])
                if nodata_value is not None:
                    rband.SetNoDataValue(nodata_value)
        else:
            rband = dataset.GetRasterBand(1)
            rband.WriteArray(in_array)
            if nodata_value is not None:
                rband.SetNoDataValue(nodata_value)

        return dataset

    def to_gdalGrid(self) -> GDALGrid:
        return GDALGrid( self.to_gdal())

    def to_tif(self, file_path: str ):
        gdalGrid = self.to_gdalGrid()
        gdalGrid.to_tif( file_path )


