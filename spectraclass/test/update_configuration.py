from spectraclass.data.base import DataManager
from spectraclass.features.texture.manager import TextureManager, texm

dm: DataManager = DataManager.initialize( "demo2", 'aviris' )
texm = texm()
dm.save_config()