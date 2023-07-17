import pickle, random, time, numpy as np
from spectraclass.data.base import DataManager
# import tensorflow as tf
# keras = tf.keras
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
encoder: Model = None
model_dims = 16
niter = 100
vae = False
key = "0000"

for image_index in range(dm.modal.num_images):
    dm.modal.set_current_image(image_index)
    print( f"Preprocessing data blocks{tm().block_dims} for image {dm.modal.image_name}" )
    base_file_path = f"{dm.cache_dir}/{dm.modal.image_name}"
    blocks = tm().tile.getBlocks()
    nblocks = len( blocks )

    initial_epoch = 0
    for iter in range(niter):
        for bi, block in enumerate(blocks):
            t0 = time.time()
            point_data, grid = block.getPointData()
            if point_data.shape[0] > 0:
                norm_data = pnorm( point_data )
                print(f"\nITER[{iter}]: Processing block{block.block_coords}, data shape = {point_data.shape}")
                if autoencoder is None:
                    autoencoder, encoder, prebuilt = rm().get_trained_network(norm_data.shape[1], model_dims, vae=vae)
                nepochs = iter*nblocks + bi
                autoencoder.fit( norm_data.data, norm_data.data, initial_epoch=nepochs-1, epochs=nepochs, shuffle=True )
            block.initialize()

    autoencoder.save( f"{base_file_path}.autoencoder.{key}" )
    encoder.save(f"{base_file_path}.encoder.{key}")
    print( f"Completed, saved model to '{base_file_path}.*encoder.{key}'")









