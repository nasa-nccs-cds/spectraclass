import pickle, random, time, numpy as np
from spectraclass.data.base import DataManager
from tensorflow.keras.models import Model
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.reduction.embedding import rm
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Optional, Dict, Tuple
import os, xarray as xa

def pnorm( data: xa.DataArray, dim: int = 1 ) -> xa.DataArray:
    dave, dmag = np.nanmean(data.values, keepdims=True, axis=dim), np.nanstd(data.values, keepdims=True, axis=dim)
    normed_data = (data.values - dave) / dmag
    return data.copy(data=normed_data)

dm: DataManager = DataManager.initialize( 'img_mgr', "aviris" )
autoencoder: Model = None
block: Block = None
encoder: Model = None
model_dims = 24
nepochs = 1
niter = 2
key = "0000"

for image_index in range(dm.modal.num_images):
    dm.modal.set_current_image(image_index)
    print( f"Preprocessing data blocks{tm().block_dims} for image {dm.modal.image_name}" )

    for iter in range(niter):
        for block in tm().tile.getBlocks():
            t0 = time.time()
            point_data, grid = block.getPointData()
            if point_data.shape[0] > 0:
                norm_data = pnorm( point_data )
                print(f"\nITER[{iter}]: Processing block{block.block_coords}, data shape = {point_data.shape}")
                if autoencoder is None:
                    autoencoder, encoder, prebuilt = rm().get_network( norm_data.shape[1], model_dims )
                autoencoder.fit( norm_data.data, norm_data.data, epochs=nepochs, batch_size=256, shuffle=True )
                print(f" Trained autoencoder in {time.time()-t0} sec")
            block.initialize()

    autoencoder.save( f"{dm.cache_dir}/autoencoder.{model_dims}.{key}" )
    encoder.save(     f"{dm.cache_dir}/encoder.{model_dims}.{key}" )
    print( f"Completed, saved model to '{dm.cache_dir}/*encoder.{key}'")









