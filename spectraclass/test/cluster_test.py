from spectraclass.learn.cluster.manager import clm
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.modes import AvirisDataManager
import xarray as xa

model_dims = 32
block_size = 150
nepochs = 5
niter = 2
method = "vae"

dm: DataManager = DataManager.initialize("img_mgr", 'aviris')
TileManager.block_size = block_size
AvirisDataManager.model_dims = model_dims
AvirisDataManager.reduce_nepoch = nepochs
AvirisDataManager.reduce_niter = niter
AvirisDataManager.reduce_method = method
AvirisDataManager.modelkey = f"b{block_size}.{method}"

print(f"Creating clusters using {clm().mid}... ")
cluster_input: xa.DataArray = dm.getModelData()
cluster_image: xa.DataArray = clm().cluster(cluster_input)