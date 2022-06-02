import pickle, random, time, numpy as np
from spectraclass.data.base import DataManager
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.data.spatial.tile.manager import TileManager, tm
from typing import List, Optional, Dict, Tuple
import os, xarray as xa

def get_spatial_norm( point_data: xa.DataArray ):
    t0 = time.time()
    norm_data = {}
    for iBand in range( point_data.shape[1] ):
        band_array: xa.DataArray = point_data[:,iBand].squeeze()
        band_data: np.ndarray = band_array.to_numpy().flatten()
        if band_data.size > 0:
            norm_data[iBand] = ( np.nansum(band_data), np.nanvar(band_data), band_data.size-1 )
    print( f"Computed block spatial_norm in {(time.time()-t0)} sec.")
    return norm_data

def pnorm( data: xa.DataArray, dim: int = 1 ) -> xa.DataArray:
    dave, dmag = np.nanmean(data.values, keepdims=True, axis=dim), np.nanstd(data.values, keepdims=True, axis=dim)
    normed_data = (data.values - dave) / dmag
    return data.copy(data=normed_data)

dm: DataManager = DataManager.initialize( 'demo2', "aviris" )

for image_index in range(dm.modal.num_images):
    dm.modal.set_current_image(image_index)
    print(f"Preprocessing data blocks for image {dm.modal.image_name}", "blue")
    norm_file_path = f"{dm.cache_dir}/{dm.modal.image_name}.norm.pkl"
    norm_data = {}

    for block in tm().tile.getBlocks():
        print(f" Processing block{block.block_coords}")
        point_data, grid = block.getPointData()
        if point_data.shape[0] > 0:
             norm_data[block.block_coords] = get_spatial_norm( point_data )

    a_file = open( norm_file_path, "wb" )
    print(f" Saving results to file '{norm_file_path}'")
    pickle.dump( norm_data, a_file )
    a_file.close()






