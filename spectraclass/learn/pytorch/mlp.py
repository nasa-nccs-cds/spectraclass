import os
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from collections import OrderedDict

from functools import partial
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import torch, math
from torch import Tensor, nn
import xarray as xa, numpy as np
from enum import Enum

def tsum(t: Tensor):
    return torch.sum(t, dim=0, keepdim=False)

def crange( data: xa.DataArray, idim:int ) -> str:
    sdim = data.dims[idim]
    c: np.ndarray = data.coords[sdim].values
    return f"[{c.min():.2f}, {c.max():.2f}]"

class ProcessingStage(Enum):
    PreTrain = 0
    Training = 1
    PostTrain = 2
    Attribution = 3
    Evaluation = 4

class MLP(nn.Module):

    def __init__(self, input_dims: int, nclasses: int, **kwargs) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.nclasses = nclasses
        self._layer_outputs: Dict[int, List[np.ndarray]] = {}
        self._layer_weights: Dict[int, List[np.ndarray]] = {}
        self._activation = kwargs.get('activation', 'lru')
        self._actparm = kwargs.get( 'actparm', 0.01 )
        self._stage = ProcessingStage.PreTrain
        self._network: nn.Sequential = None
        self._l1_strength: np.ndarray = None
        self._l2_strength: np.ndarray = None
        self.init_bias = kwargs.get('init_bias', 0.01 )
        self.wmag = kwargs.get('wmag', 0.01 )
        self.nLayers = 0
        self._L0 = 0.0
        self._iteration = 0
        self._log_step = kwargs.get( 'log_step', 2 )

    def weights_init_uniform_rule(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def weights_init_xavier(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight) # , self.wmag)
            m.bias.data.fill_(0)
 #           torch.nn.init.uniform_(m.bias, -self.init_bias, self.init_bias)

    @exception_handled
    def build_model(self, layer_sizes, **kwargs):
        lgm().log(f"#MPL: BUILD NETWORK: {self.input_dims} -> {self.nclasses}")
        in_features, iLayer = self.input_dims, 0
        self._network = nn.Sequential()
        for iL, layer_size in enumerate(layer_sizes):
            linear = nn.Linear(in_features=in_features, out_features=layer_size, bias=True)
            linear.register_forward_hook(partial(self.layer_forward_hook, iL))
            linear.register_forward_pre_hook(partial(self.layer_forward_pre_hook, iL))
            self._network.append( linear )
            self._network.append( self._activation )
            in_features, iLayer = layer_size, iLayer + 1
        linear = nn.Linear(in_features=in_features, out_features=self.nclasses, bias=True)
        self._network.append( linear )
        self.init_weights()

    def init_weights(self):
        self._network.apply(self.weights_init_uniform_rule)

    def layer_forward_hook(self, iLayer: int, module: nn.Linear, inputs: Tuple[Tensor], output: Tensor):
        if (self._stage == ProcessingStage.Training) and ( self._iteration % self._log_step == 0 ):
            fwts: np.ndarray = module.weight.detach().numpy().copy()
            self._layer_weights.setdefault(iLayer, []).append(fwts)
            #            if iLayer == 0: print( f"\nE[{len(self._layer_weights[0])}] wtsamples:  ", end="" )
            #            print( f"L{iLayer}-{fwts[0][:5]} ", end="")
            self._layer_outputs.setdefault(iLayer, []).append(output.detach().numpy().copy())

    def layer_forward_pre_hook(self, iLayer: int, module: nn.Linear, inputs: Tuple[Tensor]):
        self._iteration = self._iteration + 1
        # lgm().log(f" ** layer_forward_pre_hook[{self._iteration}][{iLayer}]: in_features={module.in_features}, "
        #              f"out_features={module.out_features}, input_shapes={[tuple(input.size()) for input in inputs]}")

    def get_layer_weights(self, iLayer: int) -> np.ndarray:
        return np.stack(self._layer_weights[iLayer], axis=1)

    def get_layer_output(self, iLayer) -> np.ndarray:
        return np.stack(self._layer_outputs[iLayer], axis=1)

    @property
    def class_layer_index(self):
        return len(self._layer_outputs) - 2

    @property
    def top_layer_index(self):
        return len(self._layer_outputs) - 1

    def get_classes(self, thresholded: bool = True) -> List[np.array]:
        classes: List[np.array] = self._layer_outputs[self.feature_layer_index]
        act = self._get_activation_function()
        if thresholded: classes = [act(torch.from_numpy(f)).detach().numpy() for f in classes]
        return classes

    def get_class_weights(self) -> np.array:
        return self.get_layer_weights(self.top_layer_index)

    def _get_activation_function(self, activation: str = None) -> nn.Module:
        if activation is None: activation = self._activation
        act = activation.lower()
        if act == "relu":
            return nn.ReLU()
        elif act == "lru":
            return nn.LeakyReLU( negative_slope=self._actparm )
        elif act == "celu":
            return nn.CELU(self._actparm)
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "logsigmoid":
            return nn.LogSigmoid()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "softmax":
            return nn.Softmax()

    def training_step_end(self, step_output):
        return step_output

    def train(self, mode: bool = True):
        self._stage = ProcessingStage.Training if mode else ProcessingStage.Evaluation
        return nn.Module.train(self, mode)

    def predict(self, data: xa.DataArray) -> xa.DataArray:
        input: Tensor = torch.from_numpy(data.values)
        result: np.ndarray = self.forward(input).detach().numpy()
        return xa.DataArray(result, dims=['samples', 'y'], coords=dict(samples=data.coords['samples'], y=range(result.shape[1])), attrs=data.attrs)

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.network(x)
        return result

    @property
    def network_type(self) -> str:
        return f"{self.input_dims}-{self.nclasses}"

    @property
    def results_dir(self):
        from spectraclass.data.base import DataManager, dm
        return dm().cache_dir

    def save(self, name: str):
        models_dir = f"{self.results_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        try:
            model_path = f"{models_dir}/{name}.{self.network_type}.pth"
            torch.save(self._network.state_dict(), model_path)
            print(f"Saved network to file '{model_path}'" )
        except Exception as err:
            print(f"Error saving model {name}: {err}")

    @property
    def network( self ) -> nn.Sequential:
        if self._network is None:
            self.build_model()
        return self._network

    def load_weights(self, filepath: str ):
        weights = torch.load(filepath)
        self.network().load_state_dict(weights)
        print(f"Loaded weights from file '{filepath}'")

    def load(self, name: str, **kwargs) -> bool:
        models_dir = f"{self.results_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        try:
            model_path = f"{models_dir}/{name}.{self.network_type}.pth"
            self.load_weights( model_path )
        except Exception as err:
            lgm().log(f"Error loading model {name} (model_path):\n  ---> {err}")
            return False
        self.eval()
        lgm().log(f"Loaded MODEL {name}: {model_path}")
        return True

    def get_learning_metrics(self):
        metrics = {}
        for mid in ['C', 'N', 'L', 'result']:
            mval = self.get_metric_values(mid)
            if mval is not None: metrics[mid] = mval
        return metrics

    def pdist(self, y: Tensor) -> Tensor:
        z = torch.exp(y)
        N = torch.sum(z, dim=0)
        return z / N

    def enorm(self, yhat: Tensor, E: Tensor, tau: float = 0.5):
        z = tau * torch.exp(yhat - E)
        return z / (1 - tau + z)

    @classmethod
    def logistic(cls, P: Tensor, tau: float = 1.0) -> Tensor:
        return tau * torch.exp(cls.entropy(P)) * P
