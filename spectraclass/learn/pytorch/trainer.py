from typing import List, Union, Tuple, Optional, Dict, Type, Callable
import torch, time, os
import traitlets as tl
from spectraclass.learn.pytorch.progress import ProgressPanel
from spectraclass.gui.control import ufm
from spectraclass.reduction.trainer import mt
from holoviews.streams import Stream, param
from panel.layout.base import Panel
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.tile.tile import Block
from spectraclass.data.base import DataManager, dm
from torch.nn import CrossEntropyLoss
from torch import Tensor, argmax
import xarray as xa, numpy as np
from .mlp import MLP
import holoviews as hv, panel as pn
import hvplot.xarray  # noqa

def stat( data: xa.DataArray ) -> str:
    x: np.ndarray = data.values
    return f"({np.nanmean(x):.2f}, {np.nanstd(x):.2f}, {np.count_nonzero( np.isnan(x) ):.2f})"

def crange( data: xa.DataArray, idim:int ) -> str:
    sdim = data.dims[idim]
    c: np.ndarray = data.coords[sdim].values
    return f"[{c.min():.2f}, {c.max():.2f}]"

def mpt() -> "ModelTrainer":
    return ModelTrainer.instance()

def random_sample( tensor: Tensor, nsamples: int, axis=0 ) -> Tensor:
    perm = torch.randperm( tensor.size(axis) )
    return tensor[ perm[:nsamples] ]

def anomaly( train_data: Tensor, reproduced_data: Tensor ) -> Tensor:
    return torch.sum( torch.abs(train_data - reproduced_data), 1 )

class ModelTrainer(SCSingletonConfigurable):
    optimizer_type = tl.Unicode(default_value="adam").tag(config=True, sync=True)
    learning_rate = tl.Float(0.01).tag(config=True, sync=True)
    loss_threshold = tl.Float(1e-6).tag(config=True, sync=True)
    init_wts_mag = tl.Float(0.1).tag(config=True, sync=True)
    init_bias_mag = tl.Float(0.1).tag(config=True, sync=True)
    nclasses = tl.Int(2).tag(config=True, sync=True)
    layer_sizes = tl.List( default_value=[64, 32, 8] ).tag(config=True, sync=True)
    modelkey = tl.Unicode(default_value="").tag(config=True, sync=True)
    nepoch = tl.Int(1).tag(config=True, sync=True)
    focus_nepoch = tl.Int(0).tag(config=True, sync=True)
    focus_ratio = tl.Float(10.0).tag(config=True, sync=True)
    focus_threshold = tl.Float(0.1).tag(config=True, sync=True)
    niter = tl.Int(100).tag(config=True, sync=True)
    log_step = tl.Int(10).tag(config=True, sync=True)
    refresh_model = tl.Bool(False).tag(config=True, sync=True)

    def __init__(self, **kwargs ):
        super(ModelTrainer, self).__init__()
        self.device = kwargs.get('device','cpu')
        self.previous_loss: float = 1e10
        self._model: MLP = None
        self._abort = False
        self._optimizer = None
        self.loss = CrossEntropyLoss( **kwargs )
        self._progress = None
        self.train_losses = None
        self.mask_save_panel = MaskSavePanel()
        self.mask_load_panel = MaskLoadPanel()

    def set_network_size(self, layer_sizes: List[int], nclasses: int):
        self.layer_sizes = layer_sizes
        self.nclasses = nclasses

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
            opts = dict ( wmag=self.init_wts_mag, init_bias=self.init_bias_mag, log_step=self.log_step )
            b: Block = tm().getBlock()
            lgm().log( f"MODEL: input dims={b.point_data.shape[1]}, layer_sizes={self.layer_sizes}" )
            self._model = MLP( "masks", b.point_data.shape[1], self.nclasses, self.layer_sizes, **opts ).to(self.device)
#            input: Tensor = torch.from_numpy(ptdata.values)
#            self._model.forward( input )
        return self._model

    def get_mask_load_panel(self) -> Panel:
        return self.mask_load_panel.gui()

    def get_class_mask(self, ptdata: xa.DataArray ) -> np.ndarray:
        return self.mask_load_panel.get_class_mask( ptdata )

    def panel(self)-> pn.Column:
        return pn.Column( self.progress.panel(), self.mask_save_panel.gui() )

    def abort_callback(self, event ):
        self._abort = True

    def get_training_set(self, **kwargs ) -> Tuple[np.ndarray,np.ndarray]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        from spectraclass.model.labels import LabelsManager, Action, lm
        label_data = lm().getTrainingLabels()
        training_data, training_labels = None, None
        for ( (tindex, block_coords, cid), gids ) in label_data.items():
            block = tm().getBlock( tindex=tindex, block_coords=block_coords )
            input_data, coords = block.filtered_point_data, block.point_coords
            training_mask: np.ndarray = np.isin( input_data.samples.values, gids )
            tdata: np.ndarray = input_data.values[ training_mask ]
            tlabels: np.ndarray = np.full([gids.size], cid)
            lgm().log( f"Adding training data: tindex={tindex} bindex={block_coords} cid={cid} #gids={gids.size} data.shape={tdata.shape} labels.shape={tlabels.shape} mask.shape={training_mask.shape}")
            training_data   = tdata   if (training_data   is None) else np.append( training_data,   tdata,   axis=0 )
            training_labels = tlabels if (training_labels is None) else np.append( training_labels, tlabels, axis=0 )
        lgm().log(f"SHAPES--> training_data: {training_data.shape}, training_labels: {training_labels.shape}" )
        return ( training_data, training_labels )

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

    def print_layer_stats(self, iL: int, **kwargs ):
        O: np.ndarray = self.model.get_layer_output(iL)
        W: np.ndarray = self.model.get_layer_weights(iL - 1)
        print( f" L[{iL}]: Oms{O.shape}=[{abs(O).mean():.4f}, {O.std():.4f}], Wms{W.shape}=[{abs(W).mean():.4f}, {W.std():.4f}]", **kwargs )

    def training_epoch(self, epoch: int, x: Tensor, y: Tensor, **kwargs) -> Tuple[float,Tensor,Tensor]:
        verbose = kwargs.get( 'verbose', False )
        y_hat: Tensor = self.model.forward(x)
        loss: Tensor = self.loss( y_hat, y )
        lval: float = float(loss)
        if verbose:
            print(f"Epoch[{epoch}/{self.nepoch}]: device={self.device}, loss={lval} ",end=" ")
            iL = self.model.feature_layer_index
            self.print_layer_stats( iL )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.previous_loss = lval
        return lval, x, y_hat

    def train(self, **kwargs):
        self.train_losses = []
        training_set = kwargs.pop( 'training_set', None )
        if training_set is None:
            (train_data, labels_data) = self.get_training_set(**kwargs)
        else:
            (train_data, labels_data) = training_set
        self.model.train()
        t0, initial_epoch = time.time(), 0
        for iter in range(self.niter):
            initial_epoch = self.training_iteration(iter, initial_epoch, train_data, labels_data, **kwargs)
        lgm().log( f"Trained network in {(time.time()-t0)/60:.3f} min" )

    def training_iteration(self, iter: int, initial_epoch: int, train_data: np.ndarray, labels_data: np.ndarray, **kwargs):
        [x, y] = [torch.from_numpy(tdata).to(self.device) for tdata in [train_data,labels_data]]
        final_epoch = initial_epoch + self.nepoch
        tloss = 0.0
        for epoch  in range( initial_epoch, final_epoch ):
            tloss, x, y_hat = self.training_epoch(epoch, x, y)
        lgm().log( f" ** ITER[{iter}]: norm data shape = {train_data.shape}, loss = {tloss}")
        loss_msg = f"loss[{iter}/{self.niter}]: {tloss:>4f}"
        self.progress.update( iter+1, loss_msg, tloss )
        return final_epoch

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

    def predict(self, data: xa.DataArray = None, **kwargs) -> xa.DataArray:
        block: Block = tm().getBlock(**kwargs)
        raster = kwargs.get( 'raster', "False")
        if data is None: data = block.point_data
        raw_result: xa.DataArray = self.model.predict( data )
        lgm().trace( f"#MT: predict-> input: [shape={data.shape}, stat={stat(data)}] output: [shape={raw_result.shape}, stat={stat(raw_result)}]")
        return block.points2raster( raw_result ) if raster else raw_result

    def event(self, source: str, event ):
        print( f"Processing event[{source}]: {event}")

class MaskCache(param.Parameterized):
    mask_name = param.String(default="", doc="Name of saved mask network")

    def __init__(self ):
        super(MaskCache, self).__init__()
        self.save_dir = f"{dm().cache_dir}/masks/cluster_mask"
        os.makedirs( self.save_dir, exist_ok=True )
        self._model: MLP = None
        self._model_loaded: bool = False

    def set_model(self, model: MLP):
        self._model = model

    def get_mask_name(self, file_path: str ) -> str:
        tail: str = file_path.split("__")[-1]
        return os.path.splitext(tail)[0]

    @property
    def model_id(self):
        return tm().tileid

    @property
    def model(self):
        if self._model is None:
            self._model = mpt().model
        return self._model

    @exception_handled
    def load( self, *args ):
        self.model.load( self.model_id, self.mask_name, dir=self.save_dir )
        dm().modal.update_parameter( "Cluster Mask", self.mask_name )
        self._model_loaded = True

    @exception_handled
    def save( self, *args, **kwargs ):
        self.model.save( self.model_id, self.mask_name, dir=self.save_dir )

    @exception_handled
    def get_class_mask(self, ptdata: xa.DataArray ) -> np.ndarray:
        mask_classes: xa.DataArray = mpt().predict( ptdata, raster=False )
        mask: np.ndarray = np.argmax( mask_classes.values, axis=1, keepdims=False ).astype( np.bool )
        nvalid = np.count_nonzero( mask )
        lgm().log( f"#FPDM: filter_point_data: ptdata shape={ptdata.shape}, coords={list(ptdata.coords.keys())}, stat={stat(ptdata)}")
        lgm().log( f"#FPDM: classes: shape={mask_classes.shape}, dims={mask_classes.dims};  mask[{mask.dtype}] shape = {mask.shape}, nvalid={nvalid}")
        lgm().log( f"#FPDM: classes stat={stat(mask_classes)}")
        return mask

class MaskSavePanel(MaskCache):

    def __init__(self ):
        super(MaskSavePanel, self).__init__()
        self.mask_name_input = pn.widgets.TextInput(name='Mask Name', placeholder='Give this mask a name...')
        self.mask_name_input.link(self, value='mask_name')
        self.save_button = pn.widgets.Button(name='Save Mask', button_type='success', width=150)
        self.save_button.on_click(self.save)

    def gui(self) -> Panel:
        save_panel = pn.Row(self.mask_name_input, self.save_button)
        return pn.WidgetBox( "###Save", save_panel )

class MaskLoadPanel(MaskCache):

    def __init__(self ):
        super(MaskLoadPanel, self).__init__()
        block_selection_names = [ self.get_mask_name(f) for f in os.listdir(self.save_dir) if ("__" in f) ]
        self.load_button = pn.widgets.Button(name='Load Mask', button_type='success', width=150)
        self.load_button.on_click(self.load)
        sopts = dict( name='Cluster Mask', options=block_selection_names )
        if mt().cluster_mask != "":        self.mask_name = mt().cluster_mask
        elif len(block_selection_names):   self.mask_name = block_selection_names[0]
        self.file_selector = pn.widgets.Select( value=self.mask_name, **sopts )
        self.file_selector.link(self, value='mask_name')
        self.load()

    def gui(self) -> Panel:
        load_panel = pn.Row(self.file_selector, self.load_button)
        if self.mask_name != "": self.load()
        return load_panel


