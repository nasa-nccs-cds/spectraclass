from spectraclass.model.base import SCSingletonConfigurable
from typing import List, Optional, Dict, Type
import xarray as xa
import numpy as np
from .base import TextureHandler
import traitlets as tl

def texm(): return TextureManager.instance()

class TextureManager(SCSingletonConfigurable):
    textures = tl.List( tl.Dict, [] ).tag(config=True,sync=True)
    required_texspec_attributes = ['type', 'bands']

    def __init__(self):
        super(TextureManager, self).__init__()
        self._texture_handlers: List[TextureHandler] = []
        self.configure()

    def addTextureBands( self, base_raster: xa.DataArray ) -> xa.DataArray:   #  base_raster dims: [ band, y, x ]
        if len( self._texture_handlers ) == 0: return base_raster
        texture_bands: List[np.ndarray] = []
        dims = base_raster.dims
        for tex_handler in self._texture_handlers:
            texture_bands.extend( tex_handler.compute_features( base_raster.data ) )
        extended_data: np.ndarray = np.concatenate( [base_raster.data, texture_bands], axis=0 )
        new_band_coord: np.ndarray = np.array( range(extended_data.shape[0]) )
        new_coords = { dims[0]: new_band_coord, dims[1]: base_raster.coords[dims[1]], dims[2]: base_raster.coords[dims[2]] }
        return xa.DataArray( extended_data, dims=dims, coords=new_coords, attrs=base_raster.attrs )


    def configure(self):
        from .glcm import GLCM
        from .gabor import Gabor
        from .lbp import LBP
        for texspec in self.textures:
            self.validate_texspec( texspec )
            type: str = texspec['type'].lower()
            if   type == "glcm":     self._texture_handlers.append( GLCM( **texspec )  )
            elif type == "gabor":    self._texture_handlers.append( Gabor( **texspec ) )
            elif type == "lbp":      self._texture_handlers.append( LBP( **texspec )   )
            else:     raise Exception( f"Unrecognized texture type '{type}' in texture specification: {texspec}")

    @classmethod
    def validate_texspec(cls, texspec: Dict ):
        for ra in cls.required_texspec_attributes:
            assert ra in texspec.keys(), f"Missing attribute '{ra}' in texture specification {texspec}"
