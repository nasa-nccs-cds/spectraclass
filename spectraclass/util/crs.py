import math, utm, os
from osgeo import gdal, gdalconst, ogr, osr

class CRS:

    @classmethod
    def get_utm_sref( cls, longitude: float, latitude: float ) -> osr.SpatialReference:
        utm_centroid_info = utm.from_latlon(latitude, longitude)
        zone_number, zone_letter = utm_centroid_info[2:]

        # METHOD USING SetUTM. Not sure if better/worse
        sp_ref = osr.SpatialReference()

        south_string = ''
        if zone_letter < 'N':
            south_string = ', +south'
        proj4_utm_string = ('+proj=utm +zone={zone_number}{zone_letter}'
                            '{south_string} +ellps=WGS84 +datum=WGS84 '
                            '+units=m +no_defs') \
            .format(zone_number=abs(zone_number),
                    zone_letter=zone_letter,
                    south_string=south_string)
        ret_val = sp_ref.ImportFromProj4(proj4_utm_string)
        if ret_val == 0:
            north_zone = True
            if zone_letter < 'N':
                north_zone = False
            sp_ref.SetUTM(abs(zone_number), north_zone)

        sp_ref.AutoIdentifyEPSG()
        return sp_ref
