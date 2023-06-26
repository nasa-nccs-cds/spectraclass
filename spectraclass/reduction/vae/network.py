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

def weights_init_uniform_rule(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if m.bias is not None:
            m.bias.data.fill_(0)

def weights_init_xavier(m: nn.Module):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight) # , self.wmag)
        m.bias.data.fill_(0)
#           torch.nn.init.uniform_(m.bias, -self.init_bias, self.init_bias)
class ProcessingStage(Enum):
    PreTrain = 0
    Training = 1
    PostTrain = 2
    Attribution = 3
    Evaluation = 4

class NetworkBase(nn.Module):

    def __init__(self, input_dims: int, latent_dims: int, reduction_factor: int, **kwargs):
        super(NetworkBase, self).__init__()
        self.device = kwargs.get('device','cpu')
        self._log_step = kwargs.get( 'log_step', 5 )
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.reduction_factor = reduction_factor
        self._iteration = 0
        self._stage = ProcessingStage.PreTrain
        self._layer_outputs: Dict[int, List[np.ndarray]] = {}
        self._layer_weights: Dict[int, List[np.ndarray]] = {}
        self.hidden_layers = []
        self.activation = kwargs.get( 'activation', 'lru' )
        self.build()

    def get_layer_weights(self, iLayer: int) -> np.ndarray:
        return np.stack(self._layer_weights[iLayer], axis=1)

    def get_layer_output(self, iLayer) -> np.ndarray:
        return np.stack(self._layer_outputs[iLayer], axis=1)

    @property
    def top_layer_index(self):
        return len(self._layer_outputs) - 1

    def _get_activation_function(self, activation: str = None) -> nn.Module:
        if activation is None: activation = self._activation
        act = activation.lower()
        if   act == "relu":         return nn.ReLU()
        elif act == "lru":          return nn.LeakyReLU()
        elif act == "celu":         return nn.CELU()
        elif act == "sigmoid":      return nn.Sigmoid()
        elif act == "logsigmoid":   return nn.LogSigmoid()
        elif act == "tanh":         return nn.Tanh()
        elif act == "softmax":      return nn.Softmax()

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

    def add_hidden_layer(self, ilayer: int, layer: nn.Linear, activation: str ):
        self.hidden_layers.append( layer )
        torch.nn.init.xavier_uniform_(layer.weight)
        print(f" * Add linear layer[{ilayer}]: {layer.in_features}->{layer.out_features}")
        if self.activation != "linear":
            self.hidden_layers.append( self._get_activation_function(activation) )
        layer.register_forward_hook(partial(self.layer_forward_hook, ilayer))
        layer.register_forward_pre_hook(partial(self.layer_forward_pre_hook, ilayer))

    def apply_hidden(self, x: Tensor ) -> Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return x

class VariationalEncoder(NetworkBase):

    def __init__(self, input_dims: int, latent_dims: int, reduction_factor: int, **kwargs):
        super(VariationalEncoder, self).__init__(input_dims,latent_dims, reduction_factor, **kwargs)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        self.build()

    def build(self):
        print( "Building Encoder:")
        in_features, iLayer = self.input_dims, 0
        while True:
            out_features = int(round(in_features / self.reduction_factor))
            if out_features <= self.latent_dims: break
            linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
            self.add_hidden_layer( iLayer, linear, self.activation )
            in_features, iLayer = out_features, iLayer + 1
        print(f" * Add mu/sigma layers[{iLayer}]: {in_features}->{self.latent_dims}")
        self.mu_layer    = nn.Linear(in_features, self.latent_dims)
        self.sigma_layer = nn.Linear(in_features, self.latent_dims)

    def forward(self, x):
        x = self.apply_hidden( x )
        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z

class Decoder(NetworkBase):

    def build(self):
        print( "Building Decoder:")
        in_features, iLayer = self.latent_dims, 0
        while in_features < self.input_dims:
            out_features = min( in_features * self.reduction_factor, self.input_dims )
            linear = nn.Linear( in_features=in_features, out_features=out_features, bias=True )
            activation = self.activation if (out_features != self.input_dims) else "sigmoid"
            self.add_hidden_layer(iLayer, linear, activation )
            in_features, iLayer = out_features, iLayer + 1
    def forward(self, z):
       return self.apply_hidden(z)


class VariationalAutoencoder():

    def __init__(self, input_dims: int, model_dims: int, **kwargs) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.model_dims = model_dims
        self.reduction_factor = kwargs.get("reduction_factor",2)
        self.device = kwargs.get('device', 'cpu')

        self._stage = ProcessingStage.PreTrain
        self._encoder: VariationalEncoder = VariationalEncoder( input_dims, model_dims, **kwargs ).to(self.device)
        self._decoder: Decoder = Decoder( input_dims, model_dims, **kwargs ).to(self.device)

    def encoder(self, input: Tensor ) -> Tensor:
        return self._encoder( input )

    def decoder(self, input: Tensor) ->  Tensor:
        return self._decoder( input )


    def training_step_end(self, step_output):
        return step_output

    def predict(self, data: xa.DataArray) -> xa.DataArray:
        input: Tensor = torch.from_numpy(data.values)
        result: np.ndarray = self.forward(input).detach()
        return xa.DataArray(result, dims=['samples', 'y'], coords=dict(samples=data.coords['samples'], y=range(result.shape[1])), attrs=data.attrs)

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

    def loss(self, x: Tensor, y: Tensor ) -> Tensor:
        return ((x - y) ** 2).sum() + self._encoder.kl

    def train( self ):
        self._encoder.train()
        self._decoder.train()

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

    def network( self, type: str ) -> NetworkBase:
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
        lgm().log(f"Loaded MODEL {name}: {mpaths}")
        return True


