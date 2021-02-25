import math, random, numpy as np
from typing import List, Optional, Dict, Type, Tuple

class TextureBase:

    def __init__( self, **kwargs ):
        pass

    def compute_features( self, image: np.ndarray ) -> List[np.ndarray]:  # input_data: dims = [ y, x ]
        raise NotImplementedError()