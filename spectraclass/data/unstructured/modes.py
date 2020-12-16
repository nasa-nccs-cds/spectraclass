from .manager import UnstructuredDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.base import DataManager

class SwiftDataManager(UnstructuredDataManager):
    MODE = "swift"
    METAVARS = ['tics', "camera", "chip", "dec", 'ra', 'tmag']

    def __init__(self):
        super(SwiftDataManager, self).__init__()

class TessDataManager(UnstructuredDataManager):
    MODE = "tess"
    METAVARS = ["target_names", "obsids"]

    def __init__(self):
        super(TessDataManager, self).__init__()

def register():
    DataManager.register_mode(TessDataManager)
    DataManager.register_mode(SwiftDataManager)



