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

def crange( data: xa.DataArray, idim:int ) -> str:
    sdim = data.dims[idim]
    c: np.ndarray = data.coords[sdim].values
    return f"[{c.min():.2f}, {c.max():.2f}]"

def mt() -> "ModelTrainer":
    return ModelTrainer.instance()

def random_sample( tensor: Tensor, nsamples: int, axis=0 ) -> Tensor:
    perm = torch.randperm( tensor.size(axis) )
    return tensor[ perm[:nsamples] ]

def anomaly( train_data: Tensor, reproduced_data: Tensor ) -> Tensor:
    return torch.sum( torch.abs(train_data - reproduced_data), 1 )

class ProgressPanel:

    def __init__(self, niter: int, abort_callback: Callable ):
        self._progress = pn.indicators.Progress(name='Iterations', value=0, width=400, max=niter )
        self._log = pn.pane.Markdown("Iteration: 0")
        self._abort = pn.widgets.Button(name='Abort', button_type='primary')
        self._abort.on_click( abort_callback )

    @exception_handled
    def update(self, iteration: int, message: str ):
        self._progress.value = iteration
        self._log.object = message

    def panel(self) -> pn.Row:
        return pn.Row( pn.pane.Markdown("Learning Progress:"), self._progress, self._log, self._abort )

class ModelTrainer(SCSingletonConfigurable):
    optimizer_type = tl.Unicode(default_value="adam").tag(config=True, sync=True)
    learning_rate = tl.Float(0.0001).tag(config=True, sync=True)
    loss_threshold = tl.Float(1e-6).tag(config=True, sync=True)
    init_wts_mag = tl.Float(0.1).tag(config=True, sync=True)
    init_bias_mag = tl.Float(0.1).tag(config=True, sync=True)
    reduce_nblocks = tl.Int(250).tag(config=True, sync=True)
    reduce_nimages = tl.Int(100).tag(config=True, sync=True)
    model_dims = tl.Int(3).tag(config=True, sync=True)
    modelkey = tl.Unicode(default_value="").tag(config=True, sync=True)
    nepoch = tl.Int(5).tag(config=True, sync=True)
    focus_nepoch = tl.Int(5).tag(config=True, sync=True)
    focus_ratio = tl.Float(10.0).tag(config=True, sync=True)
    focus_threshold = tl.Float(0.1).tag(config=True, sync=True)
    niter = tl.Int(25).tag(config=True, sync=True)
    log_step = tl.Int(10).tag(config=True, sync=True)
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
        self._progress = None

    @property
    def progress(self) -> ProgressPanel:
        if self._progress is None:
            self._progress = ProgressPanel( self.niter, self.abort_callback )
        return self._progress

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
            opts = dict ( wmag=self.init_wts_mag, init_bias=self.init_bias_mag, log_step=self.log_step )
            self._model = Autoencoder( point_data.shape[1], self.nfeatures, **opts ).to(self.device)
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
        if self.refresh_model:
            lgm().log( "REFRESH MODEL")
            return False
        return self.model.load( modelId )

    def save(self, **kwargs):
        model_id = kwargs.get('id', dm().dsid() )
        self.model.save( model_id )

    def print_layer_stats(self, iL: int, **kwargs ):
        O: np.ndarray = self.model.get_layer_output(iL)
        W: np.ndarray = self.model.get_layer_weights(iL - 1)
        print( f" L[{iL}]: Oms{O.shape}=[{abs(O).mean():.4f}, {O.std():.4f}], Wms{W.shape}=[{abs(W).mean():.4f}, {W.std():.4f}]", **kwargs )

    def training_step(self, epoch: int, x: Tensor, **kwargs) -> Tuple[float,Tensor,Tensor]:
        verbose = kwargs.get( 'verbose', False )
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
        return lval, x, y_hat

    def train(self, **kwargs):
        if not self.load(**kwargs):
            self.model.train()
            t0, initial_epoch = time.time(), 0
            for iter in range(self.niter):
                initial_epoch = self.general_training(iter, initial_epoch, **kwargs )
            lgm().log( f"Trained autoencoder in {(time.time()-t0)/60:.3f} min", print=True )
            self.save(**kwargs)

    def reduce(self, data: xa.DataArray ) -> Tuple[xa.DataArray,xa.DataArray]:
        reduced: Tensor = self.model.encode( data.values, detach=False )
        reproduction: np.ndarray = self.model.decode( reduced )
        xreduced = xa.DataArray(reduced, dims=['samples', 'features'], coords=dict(samples=data.coords['samples'], features=range(reduced.shape[1])), attrs=data.attrs)
        return xreduced, data.copy( data=reproduction )

    def general_training(self, iter: int, initial_epoch: int, **kwargs ):
        from spectraclass.data.base import DataManager, dm
        from spectraclass.data.spatial.tile.tile import Block, Tile
        num_reduce_images = min( dm().modal.num_images, self.reduce_nimages )
        losses, tloss = [], 0.0
        y_hat: Tensor = None
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
                        input_tensor: Tensor = torch.from_numpy(norm_point_data.values)
                        x = input_tensor.to(self.device)
                        final_epoch = initial_epoch + self.nepoch
                        for epoch  in range( initial_epoch, final_epoch ):
                            tloss, x, y_hat = self.training_step( epoch, x )
                            losses.append( tloss )
                        lgm().log( f" ** ITER[{iter}]: Processed block{block.block_coords}, norm data shape = {norm_point_data.shape}, losses = {losses[-self.nepoch:]}")
                        initial_epoch = final_epoch
                        if self.focus_nepoch > 0:
                            final_epoch = initial_epoch + self.focus_nepoch
                            for epoch  in range( initial_epoch, final_epoch ):
                                tloss, x, y_hat = self.focused_training_step( x, y_hat )
                                losses.append( tloss )
                            lgm().log( f" ** ITER[{iter}]: Focus-processed block{block.block_coords}, norm data shape = {norm_point_data.shape}, losses = {losses[-self.focus_nepoch:]}")
                            initial_epoch = final_epoch
                    block.initialize()
        loss_msg = f"loss[{iter}/{self.niter}]: {mean(losses):>7f}"
        lgm().log( loss_msg, print=True )
        self.progress.update( iter, loss_msg )
        return initial_epoch

    def focused_training_step(self, train_input: Tensor, y_hat: Tensor ) -> Tuple[float,Tensor,Tensor]:
        x = self.get_focused_traindata( train_input, y_hat )
        y_hat: Tensor = self.model.forward(x)
        loss: Tensor = self.loss(y_hat, x)
        lval: float = float(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.previous_loss = lval
        return lval, x, y_hat
    #
    #     from spectraclass.data.base import DataManager, dm
    #     from spectraclass.data.spatial.tile.tile import Block, Tile
    #     nepoch: int = kwargs.get( 'nepoch', self.focus_nepochn )
    #     anom_focus: float = kwargs.get( 'anom_focus', self.reduce_anom_focus )
    #     if (anom_focus == 0.0) or (nepoch==0): return False
    #
    #     anomalies = {}
    #     num_reduce_images = min(dm().modal.num_images, self.reduce_nimages)
    #     for image_index in range(num_reduce_images):
    #         dm().modal.set_current_image(image_index)
    #         blocks: List[Block] = tm().tile.getBlocks()
    #         num_training_blocks = min(self.reduce_nblocks, len(blocks))
    #         lgm().log(f"Autoencoder focused training: {num_training_blocks} blocks for image[{image_index}/{num_reduce_images}]: {dm().modal.image_name}", print=True)
    #         for iB, block in enumerate(blocks):
    #             if iB < self.reduce_nblocks:
    #                 point_data, grid = block.getPointData()
    #                 if point_data.shape[0] > 0:
    #                     reproduced_data: np.ndarray = self._autoencoder.predict( point_data.values )
    #                     anomalies[(image_index,iB)] = self.get_anomaly( point_data.data, reproduced_data )
    #     full_anomaly: np.ndarray = np.concatenate( list(anomalies.values()) )
    #     t = self.get_anomaly_threshold(full_anomaly, anom_focus)
    #     lgm().log(f"autoencoder focus({anom_focus}) training: anomaly threshold = {t}", print=True)
    #     focused_datsets = []
    #     for image_index in range(num_reduce_images):
    #         dm().modal.set_current_image(image_index)
    #         blocks: List[Block] = tm().tile.getBlocks()
    #         for iB, block in enumerate(blocks):
    #             if iB < self.reduce_nblocks:
    #                 point_data, grid = block.getPointData()
    #                 if point_data.shape[0] > 0:
    #                     anomaly = anomalies[(image_index,iB)]
    #                     focused_point_data = self.get_focused_dataset(point_data.data, anomaly, t )
    #                     focused_datsets.append( focused_point_data )
    #                     ntrainsamples = nsamples( focused_datsets )
    #                     lgm().log(f" --> BLOCK[{image_index}:{block.block_coords}]: ntrainsamples = {ntrainsamples}", print=True)
    #                     if ntrainsamples > point_data.shape[0]:
    #                         focused_training_data = np.concatenate( focused_datsets )
    #                         lgm().log( f" --> Focused Training with #samples = {ntrainsamples}", print=True)
    #                         history: tf.keras.callbacks.History = self._autoencoder.fit( focused_training_data, focused_training_data, initial_epoch=initial_epoch,
    #                                                                   epochs=initial_epoch + nepoch, batch_size=256, shuffle=True)
    #                         initial_epoch = initial_epoch + nepoch
    #                         focused_datsets = []
    #     ntrainsamples = nsamples( focused_datsets )
    #     if ntrainsamples > 0:
    #         focused_training_data = np.concatenate( focused_datsets )
    #         lgm().log(f" --> Focused Training with #samples = {ntrainsamples}", print=True)
    #         history: tf.keras.callbacks.History = self._autoencoder.fit(focused_training_data, focused_training_data, initial_epoch=initial_epoch,
    #                                                  epochs=initial_epoch + nepoch, batch_size=256, shuffle=True)
    #     return initial_epoch
    #
    def get_focused_traindata(self, train_data: Tensor, y_hat: Tensor ) -> Tensor:
        anom: Tensor = anomaly( train_data, y_hat )
        amask: Tensor = (anom > self.focus_threshold)
        anom_data, std_data = train_data[amask], train_data[~amask]
        num_standard_samples = round( anom_data.shape[0]/self.focus_ratio )
        std_data_sample = std_data if (num_standard_samples >= std_data.shape[0]) else random_sample( std_data, num_standard_samples )
        return torch.cat((anom_data, std_data_sample), 0)

    def predict(self, data: xa.DataArray, **kwargs) -> xa.DataArray:
        block: Block = tm().getBlock()
        raster = kwargs.get( 'raster', "False")
        raw_result: xa.DataArray = self.model.predict( data )
        return block.points2raster( raw_result ) if raster else raw_result

    def encode(self, data: xa.DataArray, **kwargs) -> xa.DataArray:
        block: Block = tm().getBlock()
        raster = kwargs.get( 'raster', "False")
        raw_result: np.ndarray = self.model.encode( data.values )
        xresult = xa.DataArray(raw_result, dims=['samples', 'features'], coords=dict(samples=data.coords['samples'], features=range(raw_result.shape[1])), attrs=data.attrs)
        return block.points2raster( xresult ) if raster else raw_result


    def event(self, source: str, event ):
        print( f"Processing event[{source}]: {event}")
