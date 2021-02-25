import math, random, numpy as np
from typing import List, Optional, Dict, Type, Tuple

class TextureHandler:

    def __init__( self, **kwargs ):
        self.bands = kwargs.get( 'bands' )

    def compute_band_features(self, band: np.ndarray) -> List[np.ndarray]:  # band: dims = [ y, x ]
        raise NotImplementedError()

    def compute_features(self, image: np.ndarray) -> List[np.ndarray]:  # image: dims = [ band, y, x ]
        texture_bands = []
        for iBand in self.bands:
            texture_bands.extend( *self.compute_band_features( image[ iBand ] ) )
        return texture_bands