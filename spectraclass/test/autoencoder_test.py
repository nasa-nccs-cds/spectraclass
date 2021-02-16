import random, numpy as np
from spectraclass.data.base import DataManager
from spectraclass.reduction.embedding import ReductionManager, rm
import xarray as xa

dm: DataManager = DataManager.initialize("demo1",'keelin')
project_data: xa.Dataset = dm.loadCurrentProject("main")
dm.prepare_inputs()

