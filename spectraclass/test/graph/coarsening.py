from spectraclass.data.base import DataManager
from spectraclass.application.controller import app
from spectraclass.model.labels import LabelsManager, lm
import xarray as xa
from spectraclass.graph.manager import ActivationFlow, ActivationFlowManager, afm
import pygsp as gsp
from pygsp import graphs
gsp.plotting.BACKEND = 'matplotlib'

levels = 5

dm: DataManager = DataManager.initialize("demo1",'keelin')
dm.loadCurrentProject("main")
flow: ActivationFlow = afm().getActivationFlow()

graph = flow.getGraph()

G = graphs.Graph()
G.compute_fourier_basis()
Gs = gsp.reduction.graph_multiresolution(G, levels, sparsify=False)
for idx in range(levels):
    Gs[idx].plotting['plot_name'] = 'Reduction level: {}'.format(idx)
    Gs[idx].plot()

print("")