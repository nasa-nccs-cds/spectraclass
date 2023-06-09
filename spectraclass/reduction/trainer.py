from typing import List, Union, Tuple, Optional, Dict, Type, Callable
import torch, time
import traitlets as tl
from statistics import mean
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.data.base import DataManager, dm
from torch import Tensor
import xarray as xa, numpy as np
from .autoencoder import Autoencoder
import holoviews as hv, panel as pn
import hvplot.xarray  # noqa

def mt() -> "ModelTrainer":
    return ModelTrainer.instance()

class ProgressPanel:

    def __init__(self, niter: int, abort_callback: Callable ):
        self._progress = pn.indicators.Progress(name='Iterations', value=0, width=400, max=niter )
        self._log = pn.pane.Markdown("Iteration: 0")
        self._abort = pn.widgets.Button(name='Abort', button_type='primary')
        self._abort.on_click( abort_callback )

    def update(self, iteration: int, message: str ):
        self._progress.value = iteration
        self._log.object = message

    def panel(self) -> pn.Row:
        return pn.Row( pn.pane.Markdown("Learning Progress:"), self._progress, self._log, self._abort )

class ModelTrainer(SCSingletonConfigurable):
    optimizer_type = tl.Unicode(default_value="adam").tag(config=True, sync=True)
    learning_rate = tl.Float(0.0001).tag(config=True, sync=True)
    loss_threshold = tl.Float(1e-6).tag(config=True, sync=True)
    reduce_nblocks = tl.Int(250).tag(config=True, sync=True)
    reduce_nimages = tl.Int(100).tag(config=True, sync=True)
    model_dims = tl.Int(3).tag(config=True, sync=True)
    modelkey = tl.Unicode(default_value="").tag(config=True, sync=True)
    nepoch = tl.Int(1).tag(config=True, sync=True)
    niter = tl.Int(100).tag(config=True, sync=True)
    log_step = tl.Int(5).tag(config=True, sync=True)
    refresh_model = tl.Bool(False).tag(config=True, sync=True)

    def __init__(self, **kwargs ):
        super(ModelTrainer, self).__init__()
        self.device = kwargs.get('device','cpu')
        self.nfeatures = kwargs.get('nfeatures',3)
        self.previous_loss: float = 1e10
        self._model: Autoencoder = None
        self._abort = False
        self._optimizer = None
        self.loss = torch.nn.MSELoss( **kwargs )
        self.progress = ProgressPanel( self.niter, self.abort_callback )

    @property
    def optimizer(self):
        if self._optimizer == None:
            self._optimizer = self.get_optimizer()
        return self._optimizer

    @property
    def model(self):
        if self._model is None:
            block: Block = tm().getBlock()
            point_data, grid = block.getPointData()
            self._model = Autoencoder( point_data.shape[1], self.nfeatures, log_step=self.log_step ).to(self.device)
        return self._model

    def panel(self)-> pn.Row:
        return self.progress.panel()

    def abort_callback(self, event ):
        self._abort = True

    def get_optimizer(self):
        oid = self.optimizer_type
        if oid == "rmsprop":
            return torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate )
        elif oid == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate )
        elif oid == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate )
        else:
            raise Exception(f" Unknown optimizer: {oid}")


    def load(self, **kwargs ) -> bool:
        modelId = kwargs.get('id', dm().dsid())
        if self.refresh_model: return False
        return self.model.load( modelId )

    def save(self, **kwargs):
        model_id = kwargs.get('id', dm().dsid() )
        self.model.save( model_id )

    def print_layer_stats(self, iL: int, **kwargs ):
        O: np.ndarray = self.model.get_layer_output(iL)
        W: np.ndarray = self.model.get_layer_weights(iL - 1)
        print( f" L[{iL}]: Oms{O.shape}=[{abs(O).mean():.4f}, {O.std():.4f}], Wms{W.shape}=[{abs(W).mean():.4f}, {W.std():.4f}]", **kwargs )

    def training_step(self, epoch: int, input_data: xa.DataArray, **kwargs) -> float:
        verbose = kwargs.get( 'verbose', False )
        input_tensor: Tensor = torch.from_numpy( input_data.values )
        x = input_tensor.to( self.device )
        y_hat: Tensor = self.model.forward(x)
        loss: Tensor = self.loss(y_hat, x)
        lval: float = float(loss)
        if verbose: print(f"Epoch[{epoch}/{self.nepoch}]: device={self.device}, loss={lval} ",end=" ")

        if (abs(lval)<self.loss_threshold) and ( abs(lval-self.previous_loss) < self.loss_threshold ):
            self.model.init_weights()
            print( f"Reinit & restart: epoch={epoch}" )
        else:
            if verbose:
                iL = self.model.feature_layer_index
                self.print_layer_stats( iL )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.previous_loss = lval
        return lval

    def train(self, **kwargs):
        if not self.load(**kwargs):
            t0, initial_epoch = time.time(), 0
            for iter in range(self.niter):
                initial_epoch = self.general_training(iter, initial_epoch, **kwargs )
            lgm().log( f"Trained autoencoder in {(time.time()-t0)/60:.3f} min", print=True )
            self.save(**kwargs)

    def reduce(self, data: xa.DataArray ) -> Tuple[xa.DataArray,xa.DataArray]:
        reduced: xa.DataArray = self.model.encode( data )
        reproduction: xa.DataArray = self.model.decode( reduced )
        return reduced, reproduction

    def general_training(self, iter: int, initial_epoch: int, **kwargs ):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.tile import Block, Tile
        num_reduce_images = min( dm().modal.num_images, self.reduce_nimages )
        self.model.train()
        losses = []
        for image_index in range( num_reduce_images ):
            dm().modal.set_current_image(image_index)
            blocks: List[Block] = tm().tile.getBlocks()
            num_training_blocks = min( self.reduce_nblocks, len(blocks) )
            lgm().log(f"Autoencoder general training: {num_training_blocks} blocks for image[{image_index}/{num_reduce_images}]: {dm().modal.image_name}")
            lgm().log(f" NBLOCKS = {self.reduce_nblocks}/{len(blocks)}, block shape = {blocks[0].shape}")
            for iB, block in enumerate(blocks):
                if iB < self.reduce_nblocks:
                    norm_point_data, grid = block.getPointData( norm=True )
                    if norm_point_data.shape[0] > 0:
                        final_epoch = initial_epoch + self.nepoch
                        lgm().log( f" ** ITER[{iter}]: Processing block{block.block_coords}, norm data shape = {norm_point_data.shape}")
                        for epoch  in range( initial_epoch, final_epoch ):
                            tloss: float = self.training_step( epoch, norm_point_data )
                            losses.append( tloss )
                        initial_epoch = final_epoch
                    block.initialize()
        loss_msg = f"loss[{iter}/{self.niter}]: {mean(losses):>7f}"
        lgm().log( loss_msg, print=True )
        self.progress.update( iter, loss_msg )
        return initial_epoch

    def predict(self, data: xa.DataArray, **kwargs) -> xa.DataArray:
        block: Block = tm().getBlock()
        raster = kwargs.get( 'raster', "False")
        raw_result: xa.DataArray = self.model.predict( data )
        return block.points2raster( raw_result ) if raster else raw_result

    def encode(self, data: xa.DataArray, **kwargs) -> xa.DataArray:
        block: Block = tm().getBlock()
        raster = kwargs.get( 'raster', "False")
        raw_result: xa.DataArray = self.model.encode( data )
        return block.points2raster( raw_result ) if raster else raw_result


    def event(self, source: str, event ):
        print( f"Processing event[{source}]: {event}")
