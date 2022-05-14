from spectraclass.gui.spatial.aviris.manager import AvirisDatasetManager
import xarray as xa

location = "desktop"
if location == "adapt":
    mparms = dict ( cache_dir = "/adapt/nobackup/projects/ilab/cache",
                   data_dir = "/css/above/daac.ornl.gov/daacdata/above/ABoVE_Airborne_AVIRIS_NG/data/" )
elif location == "desktop":
    mparms = dict ( cache_dir = "/Volumes/Shared/Cache",
                   data_dir = "/Users/tpmaxwel/Development/Data/Aviris" )
else: raise Exception( f"Unknown location: {location}")

widget = AvirisDatasetManager( **mparms )
widget.add_block_selection()
transformed_data: xa.DataArray = widget.overlay_image_data()
drange = widget.get_color_bounds(transformed_data)
print( drange )