import os
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from collections import OrderedDict
import torch, math
from functools import partial
from spectraclass.util.logs import LogManager, lgm
from torch import Tensor, nn
import xarray as xa, numpy as np
from enum import Enum

def tsum(t: torch.Tensor):
    return torch.sum(t, dim=0, keepdim=False)

class ProcessingStage(Enum):
    PreTrain = 0
    Training = 1
    PostTrain = 2
    Attribution = 3
    Evaluation = 4

class Autoencoder(nn.Module):

    def __init__(self, input_dims: int, model_dims: int, **kwargs) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.model_dims = model_dims
        self._layer_outputs: Dict[int, List[np.ndarray]] = {}
        self._layer_weights: Dict[int, List[np.ndarray]] = {}
        self._netlayers: Dict[int, nn.Module] = {}
        self._modules: OrderedDict = OrderedDict()
        self._activation = kwargs.get('activation', 'celu')
        self._actparm = kwargs.get('actparm', 1.0)
        self._stage = ProcessingStage.PreTrain
        self._network = None
        self._encoder = None
        self._l1_strength: np.ndarray = None
        self._l2_strength: np.ndarray = None
        self.init_bias = kwargs.get('init_bias',0.0)
        self.wmag = kwargs.get('wmag', 0.0)
        self.nLayers = 0
        self._L0 = 0.0
        self.build_ae_model()

    @property
    def output_dim(self):
        return self._n_species

    @property
    def network(self) -> nn.Sequential:
        return self._network

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
            torch.nn.init.xavier_normal_(m.weight, self.wmag)
            torch.nn.init.uniform_(m.bias, -self.init_bias, self.init_bias)

    def build_ae_model(self, **kwargs):
        lgm().log(f"#AEC: RM BUILD AEC NETWORK: {self.input_dims} -> {self.model_dims}")
        reduction_factor = 2
#        dargs = dict( kernel_initializer=tf.keras.initializers.RandomNormal(stddev=winit), bias_initializer=tf.keras.initializers.Zeros() )
        in_features, iLayer = self.input_dims, 0
        while in_features > self.model_dims:
            out_features = max( int(round(in_features / reduction_factor)), self.model_dims )
            linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
            self._add_layer( iLayer, linear, self._activation )
            in_features, iLayer = out_features, iLayer
        self._encoder = nn.Sequential(self._modules)
        while in_features < self.input_dims:
            out_features = min( in_features * reduction_factor, self.input_dims )
            linear = nn.Linear( in_features=in_features, out_features=out_features, bias=True )
            self._add_layer( iLayer, linear, self._activation )
            in_features = out_features
        self._network = nn.Sequential(self._modules)

    def init_weights(self):
        self._network.apply(self.weights_init_xavier)

    def _add_layer(self, ilayer: int, layer: nn.Linear, activation: str):
        self._netlayers[ilayer] = layer
        self._modules[f"layer-{ilayer}"] = layer
        print(f" * Add linear layer[{ilayer}]: {layer.in_features}->{layer.out_features}")
        if activation != "linear":
            print(f"   ---> Add activation: {activation}")
            self._modules[f"activation-{ilayer}"] = self._get_activation_function(activation)
        layer.register_forward_hook(partial(self.layer_forward_hook, ilayer))
        layer.register_forward_pre_hook(partial(self.layer_forward_pre_hook, ilayer))

    def get_prior(self) -> torch.Tensor:
        return torch.from_numpy(self.trainer.prior.values)

    def layer_forward_hook(self, iLayer: int, module: nn.Linear, inputs: Tuple[Tensor], output: Tensor):
        if self._stage == ProcessingStage.Training:
            fwts: np.ndarray = module.weight.detach().numpy().copy()
            self._layer_weights.setdefault(iLayer, []).append(fwts)
            #            if iLayer == 0: print( f"\nE[{len(self._layer_weights[0])}] wtsamples:  ", end="" )
            #            print( f"L{iLayer}-{fwts[0][:5]} ", end="")
            self._layer_outputs.setdefault(iLayer, []).append(output.detach().numpy().copy())

    def layer_forward_pre_hook(self, iLayer: int, module: nn.Linear, inputs: Tuple[Tensor]):
        lgm().log(f" ** layer_forward_pre_hook[{iLayer}]: in_features={module.in_features}, "
                     f"out_features={module.out_features}, input_shapes={[tuple(input.size()) for input in inputs]}")

    def get_layer_weights(self, iLayer: int) -> np.ndarray:
        return np.stack(self._layer_weights[iLayer], axis=1)

    def get_layer_output(self, iLayer) -> np.ndarray:
        lIndex = iLayer if (iLayer >= 0) else self.nlayers - 1
        return np.stack(self._layer_outputs[lIndex], axis=1)

    @property
    def feature_layer_index(self):
        return len(self._layer_outputs) - 2

    @property
    def top_layer_index(self):
        return len(self._layer_outputs) - 1

    def get_features(self, thresholded: bool = True) -> List[np.array]:
        features: List[np.array] = self._layer_outputs[self.feature_layer_index]
        act = self._get_activation_function()
        if thresholded: features = [act(torch.from_numpy(f)).detach().numpy() for f in features]
        return features

    def get_feature_weights(self) -> np.array:
        return self.get_layer_weights(self.feature_layer_index)

    def get_top_weights(self) -> np.array:
        return self.get_layer_weights(self.top_layer_index)

    @property
    def nlayers(self):
        return len(self._netlayers)

    def _get_activation_function(self, activation: str = None) -> nn.Module:
        if activation is None: activation = self._activation
        act = activation.lower()
        if act == "relu":
            return nn.ReLU()
        elif act == "lru":
            return nn.LeakyReLU(self._actparm)
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
        result: np.ndarray = self.forward(input).detach()
        return xa.DataArray(result, dims=['samples', 'y'],
                            coords=dict(samples=data.coords['samples'], y=range(result.shape[1])), attrs=data.attrs)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    @property
    def network_type(self) -> str:
        return f"{self.input_dims}-{self.model_dims}"

    @property
    def results_dir(self):
        from spectraclass.data.base import DataManager, dm
        return dm().cache_dir

    def save(self, name: str) -> str:
        models_dir = f"{self.results_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = f"{models_dir}/{name}.{self.network_type}.pth"
        try:
            torch.save(self.network.state_dict(), model_path)
            print(f"Saved model to file '{model_path}'")
        except Exception as err:
            print(f"Error saving model {name}: {err}")
        return model_path

    def load(self, name: str) -> str:
        model_path = f"{self.results_dir}/models/{name}.{self.network_type}.pth"
        weights = torch.load(model_path)
        self.network.load_state_dict(weights)
        self.eval()
        print(f"Loaded model from file '{model_path}'")
        return model_path

    def get_learning_metrics(self):
        metrics = {}
        for mid in ['C', 'N', 'L', 'result']:
            mval = self.get_metric_values(mid)
            if mval is not None: metrics[mid] = mval
        return metrics

    @classmethod
    def entropy(cls, P: Tensor) -> Tensor:
        L = torch.log(P)
        return -torch.sum(P * L, dim=0)

    @classmethod
    def entropy_norm(cls, y: Tensor) -> Tensor:
        H = cls.entropy(y)
        z = torch.exp(H) * y
        return z / (1 + z)

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
