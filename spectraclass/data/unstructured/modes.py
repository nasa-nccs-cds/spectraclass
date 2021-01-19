from .manager import UnstructuredDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.data.base import DataManager

class SwiftDataManager(UnstructuredDataManager):
    from spectraclass.gui.unstructured.application import Spectraclass
    MODE = "swift"
    METAVARS = ['tics', "camera", "chip", "dec", 'ra', 'tmag']
    INPUTS = dict(embedding='scaled_specs', directory=["target_names", "obsids"], plot=dict(y="specs", x='spectra_x_axis'))
    application = Spectraclass

    def __init__(self):
        super(SwiftDataManager, self).__init__()

class TessDataManager(UnstructuredDataManager):
    from spectraclass.gui.unstructured.application import Spectraclass
    MODE = "tess"
    METAVARS = ["target_names", "obsids"]
    INPUTS = dict(embedding='scaled_lcs', directory=['tics', "camera", "chip", "dec", 'ra', 'tmag'], plot=dict(y="lcs", x='times'))
    application = Spectraclass

    def __init__(self):
        super(TessDataManager, self).__init__()

# def register():
#     DataManager.register_mode(TessDataManager)
#     DataManager.register_mode(SwiftDataManager)





