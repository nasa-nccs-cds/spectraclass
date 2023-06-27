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

class Autoencoder(nn.Module):

    def __init__(self, input_dims: int, model_dims: int, **kwargs) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.model_dims = model_dims
        self._layer_outputs: Dict[int, List[np.ndarray]] = {}
        self._layer_weights: Dict[int, List[np.ndarray]] = {}
        self._activation = kwargs.get('activation', 'lru')
        self._actparm = kwargs.get( 'actparm', 0.01 )
        self._stage = ProcessingStage.PreTrain
        self._decoder: nn.Sequential = None
        self._encoder: nn.Sequential = None
        self._l1_strength: np.ndarray = None
        self._l2_strength: np.ndarray = None
        self.init_bias = kwargs.get('init_bias', 0.01 )
        self.wmag = kwargs.get('wmag', 0.01 )
        self.nLayers = 0
        self._L0 = 0.0
        self._iteration = 0
        self._log_step = kwargs.get( 'log_step', 10 )

    @property
    def output_dim(self):
        return self._n_species

    def encoder(self, input: Tensor ) -> Tensor:
        if self._encoder is None:
            self.build_ae_model()
        return self._encoder( input )

    def decoder(self, input: Tensor) ->  Tensor:
        if self._decoder is None:
            self.build_ae_model()
        return self._decoder( input )

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
    def build_ae_model(self, **kwargs):
        lgm().log(f"#AEC: RM BUILD AEC NETWORK: {self.input_dims} -> {self.model_dims}")
        reduction_factor = 2
#        dargs = dict( kernel_initializer=tf.keras.initializers.RandomNormal(stddev=winit), bias_initializer=tf.keras.initializers.Zeros() )
        in_features, iLayer = self.input_dims, 0
        encoder_modules = OrderedDict()
        while in_features > self.model_dims:
            out_features = max( int(round(in_features / reduction_factor)), self.model_dims )
            linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
            activation = None if (out_features == self.model_dims) else self._activation
            self._add_layer( encoder_modules, iLayer, linear, activation )
            in_features, iLayer = out_features, iLayer + 1
        self._encoder = nn.Sequential(encoder_modules)
        decoder_modules = OrderedDict()
        while in_features < self.input_dims:
            out_features = min( in_features * reduction_factor, self.input_dims )
            linear = nn.Linear( in_features=in_features, out_features=out_features, bias=True )
            activation = None if (out_features==self.input_dims) else self._activation
            self._add_layer( decoder_modules, iLayer, linear, activation )
            in_features, iLayer = out_features, iLayer + 1
        self._decoder = nn.Sequential(decoder_modules)
        self.init_weights()

    def init_weights(self):
        self._encoder.apply(self.weights_init_uniform_rule)
        self._decoder.apply(self.weights_init_uniform_rule)

    def _add_layer(self, modules: OrderedDict, ilayer: int, layer: nn.Linear, activation: str):
        modules[f"layer-{ilayer}"] = layer
        print(f" * Add linear layer[{ilayer}]: {layer.in_features}->{layer.out_features}")
        if activation != "linear":
            print(f"   ---> Add activation: {activation}")
            modules[f"activation-{ilayer}"] = self._get_activation_function(activation)
        layer.register_forward_hook(partial(self.layer_forward_hook, ilayer))
        layer.register_forward_pre_hook(partial(self.layer_forward_pre_hook, ilayer))

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
        result: np.ndarray = self.forward(input).detach()
        return xa.DataArray(result, dims=['samples', 'y'],
                            coords=dict(samples=data.coords['samples'], y=range(result.shape[1])), attrs=data.attrs)

    @exception_handled
    def encode(self, data: np.ndarray, **kwargs) -> Union[np.ndarray,Tensor]:
        detach = kwargs.get('detach',False)
        input: Tensor = torch.from_numpy(data)
        result: Tensor = self.encoder(input)
        return result.detach().numpy() if detach else result

    @exception_handled
    def decode(self, data: Union[np.ndarray,Tensor]) -> np.ndarray:
        input: Tensor = torch.from_numpy(data) if (type(data) == np.ndarray) else data
        result: Tensor = self.decoder(input)
        return result.detach().numpy()

    def forward(self, x: Tensor) -> Tensor:
        encoded: Tensor = self.encoder(x)
        result = self.decoder(encoded)
        return result

    @property
    def network_type(self) -> str:
        return f"{self.input_dims}-{self.model_dims}"

    @property
    def results_dir(self):
        from spectraclass.data.base import DataManager, dm
        return dm().cache_dir

    def save(self, name: str):
        models_dir = f"{self.results_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        try:
            model_path = f"{models_dir}/{name}.encoder.{self.network_type}.pth"
            torch.save(self._encoder.state_dict(), model_path )
            print(f"Saved encoder to file '{model_path}'" )
            model_path = f"{models_dir}/{name}.decoder.{self.network_type}.pth"
            torch.save(self._decoder.state_dict(), model_path)
            print(f"Saved decoder to file '{model_path}'")
        except Exception as err:
            print(f"Error saving model {name}: {err}")

    def network( self, type: str ) -> nn.Sequential:
        if self._encoder is None:
            self.build_ae_model()
        if type == "encoder":   return self._encoder
        elif type == "decoder": return self._decoder
        else: raise Exception( f"Unlnown nnet type: {type}")

    def load_weights(self, type: str, filepath: str ):
        weights = torch.load(filepath)
        self.network(type).load_state_dict(weights)
        print(f"Loaded {type} weights from file '{filepath}'")

    def load(self, name: str, **kwargs) -> bool:
        models_dir = f"{self.results_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        mpaths = []
        try:
            for mtype in [ "encoder", "decoder" ]:
                model_path = f"{models_dir}/{name}.{mtype}.{self.network_type}.pth"
                mpaths.append( model_path )
                self.load_weights( mtype, model_path )
        except Exception as err:
            lgm().log(f"Error loading model {name} ({mpaths[-1]}):\n  ---> {err}")
            return False
        self.eval()
        lgm().log(f"Loaded MODEL {name}: {mpaths}")
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
