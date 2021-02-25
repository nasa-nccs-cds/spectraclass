import os, time, random, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from .base import TextureHandler
from skimage.feature import texture as sktex
from skimage.morphology import disk, ball
from skimage.filters.rank import windowed_histogram
from spectraclass.test.texture.util import *


class LBP(TextureHandler):

    def __init__(self, **kwargs):
        super(LBP, self).__init__( **kwargs )
        self.R = kwargs.get( 'R', 1.0 )
        self.method = kwargs.get('method', 'uniform')  # "var" "ror" "uniform"
        self.hist_radius = kwargs.get( 'hist_radius', 5 )
        self.n_epochs = kwargs.get('epochs', 100)
        self.nFeatures = kwargs.get( 'nfeat', 1 )

    def compute_band_features(self, image_band: np.ndarray) -> List[np.ndarray]:  # input_data: dims = [ y, x ]
        t0 = time.time()
        P = round( 8 * self.R )
        lbp: np.ndarray = sktex.local_binary_pattern( image_band.data, P, self.R, self.method ).astype(np.int)
        print( f" calculated LBP in time {time.time()-t0} sec, shape = {lbp.shape}, range = [ {lbp.min()}, {lbp.max()} ] ")

        t1 = time.time()
        hist_array: np.ndarray = windowed_histogram( lbp, disk( self.hist_radius )  )
        [ys,xs,nbin] = hist_array.shape
        sample_probabilties: np.ndarray = hist_array.reshape( [ys*xs, nbin] )
        print( f" calculated hist_array in time {time.time()-t1} sec, shape = {hist_array.shape}, norm[:10] = {sample_probabilties.sum(axis=1)[:10]}")

        t1 = time.time()
        ( tex, rep ) = autoencoder_reduction( sample_probabilties, self.nFeatures, self.n_epochs  )
        texture_features = tex.reshape( [ys,xs,self.nFeatures] )
        print( f" calculated autoencoder reduction in time {time.time()-t1} sec, shape = {texture_features.shape}")
        return [ texture_features[:,:,iF] for iF in range(self.nFeatures) ]