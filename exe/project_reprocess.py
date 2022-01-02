from spectraclass.data.base import DataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
import sys

if len(sys.argv) != 4:
    print( f"Usage: {sys.argv[0]} <mode> <project> <#iter>")
else:

    mode: str = sys.argv[1]       #   e.g. 'swift', 'tess', 'desis', or 'aviris'
    project: str = sys.argv[2]    #   e.g. 'demo1', 'demo2', 'demo3', or 'demo4'
    niter: int = int(sys.argv[3])

    dm: DataManager = DataManager.initialize( project, mode )
    block_nsamples: Dict[Tuple,int] = {}
    for iter in range(niter):
        block_nsamples = dm.prepare_inputs( reprocess=True )
    dm.save_config( block_nsamples )

