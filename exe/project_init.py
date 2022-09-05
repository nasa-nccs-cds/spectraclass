from spectraclass.data.base import DataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
import sys

if len(sys.argv) != 3:
    print( f"Usage: {sys.argv[0]} <mode> <project>")
else:

    mode: str = sys.argv[1]       #   e.g. 'swift', 'tess', 'desis', or 'aviris'
    project: str = sys.argv[2]    #   e.g. 'demo1', 'demo2', 'demo3', or 'demo4'

    dm: DataManager = DataManager.initialize( project, mode )
    dm.prepare_inputs( reprocess=True )


