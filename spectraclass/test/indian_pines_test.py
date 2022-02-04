import random, numpy as np, torch
from spectraclass.ext.pynndescent import NNDescent
from spectraclass.data.base import DataManager
from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
from torch_geometric.data import Data
import xarray as xa

dm: DataManager = DataManager.initialize( "indianPines", 'aviris' )
dm.prepare_inputs()
project_data: xa.Dataset = dm.loadCurrentProject( "main" )
