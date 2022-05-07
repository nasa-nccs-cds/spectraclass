import math, time, random, numpy as np
from typing import List, Optional, Dict, Type, Tuple
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

class TextureHandler:

    def __init__( self, **kwargs ):
        self.bands = kwargs.get( 'bands' )
        self.type = kwargs.get('type')

    def compute_band_features(self, band: np.ndarray) -> List[np.ndarray]:  # band: dims = [ y, x ]
        raise NotImplementedError()

    def compute_features(self, image: np.ndarray) -> List[np.ndarray]:  # image: dims = [ band, y, x ]
        texture_layers = []
        t0 = time.time()
        lgm().log(f"Computing {self.type} texture bands for image shape {image.shape}")
        for iBand in self.bands:
            t1 = time.time()
            input_band = image[ iBand ]
            tex_layers = self.compute_band_features( input_band )
            lgm().log(f"Computed {len(tex_layers)} texture feature layers for image band {iBand} (shape={input_band.shape}) in {time.time()-t1} sec.")
            texture_layers.extend( tex_layers )
        lgm().log(f"Completed {self.type} texture computation in {time.time()-t0} sec.")
        return texture_layers