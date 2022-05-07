from gui.spatial.widgets.scrap_heap.tiles import TileManager
import xarray as xa

vindex = 2
hindex = 3
token = 'dHBtYXh3ZWw6ZEdodmJXRnpMbTFoZUhkbGJHeEFibUZ6WVM1bmIzWT06MTYzMTgxNTYzOTo2ZjRjMTZjMmRiMjE1ZGZhNGIzNGYxNGQxZmQ1YWFhODg3ZWIwOTg1'
file_name = f"MCDWD_L3_F2_NRT.A2021252.h{hindex:02}v{vindex:02}.061"
file_path = f"allData/61/MCDWD_L3_F2_NRT/Recent/{file_name}.tif"
target_url = f"https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/{file_path}"
result_dir = "/tmp"
target_file_path = f"{result_dir}/{file_path}"

TileManager.download( target_url, result_dir, token )
raster = xa.open_rasterio( target_file_path )
print( f"VR[{vindex}]: {int(float(raster.attrs['SOUTHBOUNDINGCOORDINATE']))} -> {int(float(raster.attrs['NORTHBOUNDINGCOORDINATE']))}" )
print( f"HR[{hindex}]: {int(float(raster.attrs['EASTBOUNDINGCOORDINATE']))} -> {int(float(raster.attrs['WESTBOUNDINGCOORDINATE']))}" )