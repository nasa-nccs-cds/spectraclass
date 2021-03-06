from .manager import UnstructuredDataManager
from typing import List, Union, Tuple, Optional, Dict, Callable
import traitlets as tl

class SwiftDataManager(UnstructuredDataManager):
    from spectraclass.gui.unstructured.application import Spectraclass

    obsids = tl.Unicode("obsids.pkl").tag(config=True)
    scaled_specs = tl.Unicode("scaled_specs.pkl").tag(config=True)
    specs = tl.Unicode("specs.pkl").tag(config=True)
    spectra_x_axis = tl.Unicode("spectra_x_axis.pkl").tag(config=True)
    target_names = tl.Unicode("target_names.pkl").tag(config=True)

    MODE = "swift"
    METAVARS = ["target_names", "obsids"]
    INPUTS = dict(embedding='scaled_specs', directory=METAVARS, plot=dict(y="specs", x='spectra_x_axis'))
    application = Spectraclass

    def __init__(self):
        super(SwiftDataManager, self).__init__()

class TessDataManager(UnstructuredDataManager):
    from spectraclass.gui.unstructured.application import Spectraclass
    MODE = "tess"
    METAVARS = ['tics', "camera", "chip", "dec", 'ra', 'tmag']
    INPUTS = dict(embedding='scaled_lcs', directory=METAVARS, plot=dict(y="lcs", x='times'))
    application = Spectraclass
    tics = tl.Unicode("tics.pkl").tag(config=True)
    camera = tl.Unicode("camera.pkl").tag(config=True)
    chip = tl.Unicode("chip.pkl").tag(config=True)
    dec = tl.Unicode("dec.pkl").tag(config=True)
    ra = tl.Unicode("ra.pkl").tag(config=True)
    tmag = tl.Unicode("tmag.pkl").tag(config=True)
    lcs = tl.Unicode("lcs.pkl").tag(config=True)
    times = tl.Unicode("times.pkl").tag(config=True)

    def __init__(self):
        super(TessDataManager, self).__init__()

# def register():
#     DataManager.register_mode(TessDataManager)
#     DataManager.register_mode(SwiftDataManager)





