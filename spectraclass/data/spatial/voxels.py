import time, math, os, sys, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from spectraclass.util.logs import LogManager, lgm
from matplotlib import cm
import numpy.ma as ma
import xarray as xa

def close( pt0, pt1, min_dist: float ):
    return (pt0)

class Voxelizer:

    def __init__(self, points: xa.DataArray, resolution: float  ):
        self.points = points.values
        self.resolution = resolution
        self.compute_bounds( points.values )
        self.compute_voxel_indices( points )
        self.vrange = ( self.vids.min(), self.vids.max() )
        lgm().log( f" ** compute vindices[{self.vids.shape}]--> bounds: {[self.vids.min(), self.vids.max()]}")

    def compute_bounds(self, points: np.ndarray ):
        bnds = []
        for i in range(3):
            x = points[: ,i]
            bnds.append( (x.min(), x.max()) )
        self.bounds = np.array( bnds )

    @property
    def origin(self) -> np.ndarray:
        return self.bounds[:,0].reshape([1,3])

    @property
    def range(self) -> np.ndarray:
        return ( self.bounds[:,1] - self.bounds[:,0] ).reshape([1,3])

    @property
    def nbins(self) -> np.ndarray:
        return self.range / self.resolution

    def normalize( self, points: np.ndarray ) -> np.ndarray:
        return (points - self.origin) / self.range

    def serialize(self, vid3: np.ndarray ) -> np.ndarray:
        nb = self.nbins.flatten().astype(int)
        return vid3[:,0]*nb[1]*nb[2] + vid3[:,1]*nb[2] + vid3[:,2]

    def compute_voxel_indices(self, points: xa.DataArray ):
        v3id = (self.normalize(points.values) * self.nbins).astype(int)
        self.vids = self.serialize( v3id )
        self.indices = points.samples.values

    def compute_voxel_index(self, point: np.ndarray ) -> int:
        v3id = (self.normalize(point) * self.nbins).astype(int)
        return self.serialize( v3id )[0]

    def get_pid(self, point: Tuple[float,float,float]):
        pid = -1
        npt = np.array(point).reshape([1, 3])
        vid = self.compute_voxel_index( npt )
        mask = (self.vids==vid)
        lindices = self.indices[ mask ]
        if lindices.size > 0:
            vpoints: np.ndarray = self.points[mask]
            dist: np.ndarray = np.abs( vpoints - npt ).max(axis=1)
            pid = lindices[ dist.argmin() ]
        lgm().log(f" *** PCM.on_pick: pid={pid}, vid={vid}, point={point}")
        lgm().log(f"    PCM-->indices: size={self.indices.size}, range={[self.indices.min(),self.indices.max()]}")
        return pid

    # def pick_point(self, ray: np.ndarray, tolerance: float  ):
    #     t0 = time.time()
    #     rvids: np.ndarray = self.compute_voxel_indices( ray )
    #     for (rvid,rpt) in zip(rvids,ray):
    #         if (rvid >= self.vrange[0]) and (rvid <= self.vrange[1]):
    #             vpoints: np.ndarray = self.points[ self.vids==rvid ]
    #             if vpoints.size > 0:
    #                 dist = (vpoints - rpt).max( axis = 1 )
    #                 lgm().log( f"rpt[{rvid}]: {rpt} --> vpoints: {vpoints}, dist = {dist}" )
    #     lgm().log(f"completed pick_point in {time.time()-t0} sec")
