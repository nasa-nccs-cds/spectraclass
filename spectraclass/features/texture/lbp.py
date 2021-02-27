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
        self.Rs = kwargs.get( 'R', [ 1.0 ] )
        self.method = kwargs.get('method', 'uniform')  # "var" "ror" "uniform"
        self.hist_radius = kwargs.get( 'hist_radius', 3 )
        self.nFeatures = kwargs.get( 'nfeat', 1 )
        self.reduce_method = kwargs.get( 'reduce_method', "ica" )   # "ica" "pca"

    def compute_band_features(self, image_band: np.ndarray) -> List[np.ndarray]:  # input_data: dims = [ y, x ]
        t0 = time.time()
        sample_probabilties = []

        for R in self.Rs:
            P = round( 4 + 4 * R )
            lbp: np.ndarray = sktex.local_binary_pattern( image_band.data, P, R, self.method ).astype(np.int)
            print( f" calculated LBP in time {time.time()-t0} sec, shape = {lbp.shape}, range = [ {lbp.min()}, {lbp.max()} ] ")

            t1 = time.time()
            hist_array: np.ndarray = windowed_histogram( lbp, disk( self.hist_radius )  )
            [ys,xs,nbin] = hist_array.shape
            sp: np.ndarray = hist_array.reshape( [ys*xs, nbin] )
            print( f" calculated hist_array[R={R}] in time {time.time()-t1} sec, shape = {hist_array.shape}, norm[:10] = {sp.sum(axis=1)[:10]}")
            sample_probabilties.append(sp)

        t1 = time.time()
        accum_samples = np.concatenate( sample_probabilties, axis = 1 )
        print(f" Accumulated sample probabilties shape = {accum_samples.shape}, performing {self.reduce_method} reduction to {self.nFeatures} feature(s)")
        ( tex, rep ) = ca_reduction( accum_samples, self.nFeatures, self.reduce_method  )
        texture_features = tex.reshape( list(image_band.shape) + [ self.nFeatures ] )
        print( f" calculated reduction in time {time.time()-t1} sec, shape = {texture_features.shape}")
        return [ texture_features[:,:,iF] for iF in range(self.nFeatures) ]