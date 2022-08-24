from spectraclass.learn.cluster.manager import clm
from spectraclass.data.base import DataManager, dm
import xarray as xa

DataManager.initialize( "img_mgr", 'aviris' )
print(f"Creating clusters using {clm().mid}... ")
cluster_input: xa.DataArray = dm().getModelData()
cluster_image: xa.DataArray = clm().cluster(cluster_input)