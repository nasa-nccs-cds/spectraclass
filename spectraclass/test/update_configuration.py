from spectraclass.data.base import DataManager
from spectraclass.features.texture.manager import TextureManager, texm

dm: DataManager = DataManager.initialize( "demo1", 'keelin' )
texm = texm()
dm.save_config()