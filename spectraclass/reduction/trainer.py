
from typing import List, Union, Tuple, Optional, Dict, Type, Callable
import torch, glob
from functools import partial
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.data.spatial.tile.manager import TileManager, tm
from spectraclass.data.spatial.tile.tile import Block
from torch import Tensor
import xarray as xa, numpy as np
from .autoencoder import Autoencoder
import holoviews as hv, panel as pn
import hvplot.xarray  # noqa

class ProgressPanel:

    def __init__(self, abort_callback: Callable ):
        self._progress = pn.indicators.Progress(name='Iterations', value=0, width=400, max=cfg().learning.nepochs-1)
        self._log = pn.pane.Markdown("Iteration: 0")
        self._abort = pn.widgets.Button(name='Abort', button_type='primary')
        self._abort.on_click( abort_callback )

    def update(self, iteration: int, message: str ):
        self._progress.value = iteration
        self._log.object = message

    def panel(self) -> pn.Row:
        return pn.Row( pn.pane.Markdown("Learning Progress:"), self._progress, self._log, self._abort )

class ModelTrainer:

    def __init__(self, **kwargs ):
        self.device = kwargs.get('device','cpu')
        self.nfeatures = kwargs.get('nfeatures',3)
        self.previous_loss: float = 1e10
        block: Block = tm().getBlock()
        point_data, grid = block.getPointData()
        self._model: Autoencoder = kwargs.get( 'model', Autoencoder( point_data.shape[1], self.nfeatures ) ).to(self.device)
        self._abort = False
        self.optimizer = self.get_optimizer()
        self.loss = torch.nn.MSELoss( **kwargs )
        self.progress = ProgressPanel( self.abort_callback )


    def panel(self)-> pn.Row:
        return self.progress.panel()

    def abort_callback(self, event ):
        self._abort = True

    def get_optimizer(self):
        oid = cfg().learning.get('optimizer',"adam")
        lr = cfg().learning.learning_rate
        if oid == "rmsprop":
            return torch.optim.RMSprop(self._model.parameters(), lr=lr)
        elif oid == "adam":
            return torch.optim.Adam(self._model.parameters(), lr=lr)
        elif oid == "sgd":
            return torch.optim.SGD(self._model.parameters(), lr=lr)
        else:
            raise Exception(f" Unknown optimizer: {oid}")

    @property
    def xext(self):
        return self.dataloader.cov.xext

    @property
    def yext(self):
        return self.dataloader.cov.yext

    def get_metric_values(self, name: str) -> Optional[xa.DataArray]:
        species: List[str] = [s.name for s in self.dataloader.occ.species]
        tensors = self.loss.get_metric( name )
        arrays = np.array([t.detach().numpy() for t in tensors])
        dims = ['iterations']
        coords = dict(iterations=self.loss.metrics_iterations)
        if arrays.size > 0:
            if arrays.ndim > 1:
                dims.append('species')
                coords['species'] = species
            if arrays.ndim == 3:
                dims.insert(1, 'samples')
                coords['samples'] = range(arrays.shape[1])
            return xa.DataArray(arrays, name=name, dims=dims, coords=coords, attrs=self.dataloader.data_attributes)
        lgm().log(f"No data for metric {name}")

    def load(self, modelId: str ):
        self._model.load( modelId )

    def save(self, **kwargs):
        model_id = kwargs.get('id', cfg().scenario.id)
        self._model.save( model_id )

    def print_layer_stats(self, iL: int, **kwargs ):
        O: np.ndarray = self._model.get_layer_output(iL)
        W: np.ndarray = self._model.get_layer_weights(iL - 1)
        print( f" L[{iL}]: Oms{O.shape}=[{abs(O).mean():.4f}, {O.std():.4f}], Wms{W.shape}=[{abs(W).mean():.4f}, {W.std():.4f}]", **kwargs )

    def training_step(self, epoch: int, batch: Tuple[Tensor, Tensor], **kwargs) -> Tensor:
        verbose = kwargs.get( 'verbose', False )
        nepochs = cfg().learning.nepochs
        x, target = batch
        y_hat: Tensor = self._model.forward(x)
        loss: Tensor = self.loss(y_hat, target)
        lt = cfg().learning.loss_threshold
        lval: float = float(loss)
        if verbose: print(f"Epoch[{epoch}/{nepochs}]: loss={lval} ",end=" ")

        if (abs(lval)<lt) and ( abs(lval-self.previous_loss) < lt ):
            self._model.init_weights()
            print( f"Reinit & restart: epoch={epoch}" )
        else:
            if verbose:
                iL = self._model.feature_layer_index
                self.print_layer_stats( iL )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.previous_loss = lval
        return loss


        # # L1 regularizer
        # for iL, (layer, l1p) in enumerate(zip(self._netlayers, self._l1_strength)):
        #     if l1p > 0.0:
        #         wabs: Tensor = layer.weight.abs()
        #         l1_reg: Tensor = wabs.sum()
        #         wmax: Tensor = wabs.max()
        #         self.add_metric_value(f"wmax-{iL}", wmax)
        #         self.add_metric_value(f"l1_reg-{iL}", l1_reg)
        #         self.add_metric_value(f"l1_reg-{iL}_step", Tensor([self.global_step]))
        #         loss += l1p * l1_reg
        #
        # # L2 regularizer
        # for iL, (layer, l2p) in enumerate(zip(self._netlayers, self._l2_strength)):
        #     if l2p > 0.0:
        #         l2_reg = layer.weight.pow(2).sum()
        #         loss += l2p * l2_reg

        # tensorboard_logs = {"train_ce_loss": loss}
        # progress_bar_metrics = tensorboard_logs
        # return {"loss": loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    @log_timing
    def train(self,**kwargs):
        nepochs = cfg().learning.nepochs
        self._abort = False
        self._model.train()
        for epoch in range( nepochs ):
            if self._abort: return
            for batch, (X, y) in enumerate(self.dataloader):
                X, y = X.to(self.device), y.to(self.device)
                tloss: Tensor = self.training_step( epoch, (X,y),**kwargs )
                self.progress.update( epoch, f"loss[{epoch}/{nepochs}]: {tloss:>7f}" )
        return



    # def plot_dual_rep(self, time_index=0, **plot_args  ):
    #     date = self.dataloader.dates[time_index]
    #     input = self.dataloader.get_samples( date )
    #     Yhat: Tensor = self._model.network( torch.from_numpy(input.values) ).detach()
    #     xYhat = xa.DataArray( Yhat, dims=['samples'], coords=dict(samples=input.coords['samples']), attrs=input.attrs)
    #     print( f"xYhat range: {xYhat.values.min()} {xYhat.values.max()} {xYhat.values.mean()}")
    #     Y: Tensor = self._model.logistic(Yhat)
    #     xY = xa.DataArray( Y, dims=['samples'], coords=dict(samples=input.coords['samples']), attrs=input.attrs)
    #
    #     features: xa.DataArray = self._model.get_features(input)
    #     result_weights: np.ndarray = self._model.result_weights
    #     nF = features.shape[1]
    #
    #     fig, axs = plt.subplots(3, nF+1 )
    #
    #     self.dataloader.imshow( xYhat, axs[0,0], "Yhat", **plot_args )
    #
    #     ssum: xa.DataArray = None
    #     for iF in range(nF):
    #         feature_samples = features[:,iF]
    #         print(f"F{iF} range: {feature_samples.values.min()} {feature_samples.values.max()} {feature_samples.values.mean()}")
    #         feature_map: xa.DataArray = self.dataloader.toraster( feature_samples )
    #         self.dataloader.imshow( feature_map, axs[0,iF+1], f"Feature-{iF}", **plot_args)
    #         wF: xa.DataArray = feature_samples * result_weights[iF]
    #         ssum = wF if (ssum is None) else ssum+wF
    #     ssum.attrs.update( features.attrs )
    #     wsum: xa.DataArray = self.dataloader.toraster( ssum )
    #     print(f"wsum range: {np.nanmin(wsum.values)} {np.nanmax(wsum.values)} {np.nanmean(wsum.values)}")
    #
    #     self._model.plot_top_weights( axs[1,0] )
    #     fws: np.ndarray = self._model.feature_weights
    #     for ifw in range( fws.shape[0] ):
    #         self._model.plot_weights( fws[ifw], axs[1,ifw+1], f"F-{ifw} feature_weights", x=input.covariates.values )
    #
    #     self.dataloader.imshow( wsum, axs[2,0], f"wsum", **plot_args )
    #     self.dataloader.imshow( xY,   axs[2,1], f"Y",    **plot_args )
    #
    #     plt.show()

    def load_reference_attribution( self, covariates: List[str] ) -> Dict[str,np.ndarray]:
        cfg_ref = cfg().verification.get('reference_attribution')
        if cfg_ref:
            ref_files: List = glob.glob( cfg_ref.format(sparrow=cfg().platform.sparrow) )
            ref_data_arrays: List = [ self.preprocessor.load_csv_file( ref_file, header=False ) for ref_file in ref_files]
            if len( ref_data_arrays ) > 0:
                ref_data: np.ndarray = np.stack( [self.process_attr_data(rdata,covariates) for rdata in ref_data_arrays], axis=1 )
                return dict( mean=ref_data.mean(axis=1), max=ref_data.max(axis=1), min=ref_data.min(axis=1) )

    def process_attr_data(self, ref_attr_list: List, covariates: List[str] ) -> np.array:
        ref_attrs: Dict = dict([elem.split(',') for elem in ref_attr_list[0]])
        ref_attrs: Dict = {id.split(".")[0].strip('"'): float(sv) for id, sv in ref_attrs.items()}
        radata = np.array( [ref_attrs.get(covar.split("___")[1], 0.0) for covar in covariates] )
        return radata / radata.mean()

    def get_reference_attribution(self, samples: xa.DataArray) -> Dict[str,np.ndarray]:
        covariates: List[str] = samples.covariates.values.tolist()
        ref_attr: Dict[str, np.ndarray] = self.load_reference_attribution(covariates)
        return ref_attr

    def _xafeature(self, samples: xa.DataArray, features: np.ndarray ) -> xa.DataArray:
        return xa.DataArray( features, dims=['samples', 'features'],
                                coords=dict(samples=samples.coords['samples']), attrs=samples.attrs )

    def _rasterize_feature(self, iF: int, feature: xa.DataArray ) -> xa.DataArray:
        rf = self.dataloader.toraster( feature[:, iF] )
        rf.name = f"Feature-{iF}"
        rf.attrs['index'] = iF
        return rf


    # def mplplot_feature(self, title: str, feature: xa.DataArray, attribution: np.ndarray, **kwargs ):
    #     result_map: xa.DataArray = self.dataloader.toraster( feature )
    #     fig_height = max( 5, 0.1 * attribution.shape[0] )
    #     plot_height = round( 10.3 - 0.043 * attribution.shape[0] )
    #     fig = plt.figure(title, figsize=(15, fig_height), constrained_layout=True)
    #     fig.canvas.mpl_connect('button_press_event', partial(self.event,'feature_plot') )
    #     gs = GridSpec(9, 6, figure=fig)
    #     ax1: Axes = fig.add_subplot( gs[0:plot_height,0:-2], projection=PlateCarree() )
    #     ax1.set_extent( result_map.attrs['extent'] )
    #     ax1.add_feature(cfeature.COASTLINE, zorder=10, color="magenta" )
    #     ax1.imshow( result_map.values, cmap="jet", origin="lower", alpha=1.0, zorder=5, extent=result_map.attrs['extent']  )
    #     ax2: Axes = fig.add_subplot(gs[:,-2:])
    #     vnames: List[str] = list(self.dataloader.cov.varnames.values())
    #     ax2.barh( vnames, attribution )
    #     ax2.set_yticks( ax2.get_yticks(), labels=vnames, fontsize=8 )
    #     return fig.canvas

    # def hvplot_feature(self, title: str, feature: xa.DataArray, attribution: np.ndarray, **kwargs ):
    #     result_map: xa.DataArray = self.dataloader.toraster( feature )
    #     feature_plot = result_map.hvplot.image(x='lon', y='lat', cmap="jet", width=600, height=500)
    #     vnames: List[str] = list(self.dataloader.cov.varnames.values())
    #     attrplot = xattribution.hvplot.barh( )
    #     return pn.Row( feature_plot, attrplot )

    def get_xraster(self, samples_data: np.ndarray, **kwargs ):
        feature = kwargs.get( 'feature', -1 )
        samples = samples_data if ((feature < 0) or (samples_data.ndim == 1)) else samples_data[:,feature]
        return self.dataloader.toraster( xa.DataArray( samples, dims=['samples'], attrs=self.dataloader.grid_attrs ) )

    def plot_loss_element(self, ax, index: int):
        self._model.plot_loss_element( ax, index)

    def plot_loss_metrics(self, ax, index: int):
        self._model.plot_loss_metrics( ax, index)

    def get_occurences( self ) -> Dict[str,Dict[str,np.ndarray]]:
        return { s.name: s.points(self.xext,self.yext) for s in self.dataloader.occ.species }

    def get_species( self ) -> List[str]:
        return [ s.name for s in self.dataloader.occ.species ]

    def get_loss_metrics(self,**kwargs) -> Dict[str,xa.DataArray]:
        raster = kwargs.get('raster',True)
        metrics = { mid: self.get_metric_values(mid) for mid in [ "C", "N", "L", "loss","result"] }
        if raster: metrics = { mid: self.dataloader.toraster(mdata) for (mid,mdata) in metrics.items()}
        return metrics

    # def plot_training_elements( self, ax, element_name: str, plot_args: Dict, **kwargs ) -> ImageBrowser:
    #     metrics: xa.DataArray = self._model.get_metric_values( element_name )
    #     rasters: List[xa.DataArray] = [ self.get_xraster( metrics[step], **kwargs ) for step in range(metrics.shape[0]) ]
    #     overlays: List[str] = kwargs.get( 'overlays', [] )
    #     overlay_plots = {}
    #     for overlay_element_name in overlays:
    #         overlay: np.ndarray = self._model.get_metric_values( overlay_element_name )
    #         if overlay.ndim == 1: overlay_plots[overlay_element_name] = [ self.get_xraster(overlay) ]
    #         else: overlay_plots[overlay_element_name] = [ self.get_xraster( overlay[step], **kwargs ) for step in range(metrics.shape[0]) ]
    #     plot_args['overlays'] = overlay_plots
    #     plot_args['name'] = element_name
    #     image_browser = ImageBrowser( 'Train Step', ax, rasters, plot_args )
    #     self.image_browsers[element_name] = image_browser
    #     return image_browser

    def prediction(self, data: xa.DataArray, **kwargs) -> xa.DataArray:
        from sparrow.data.dataset import SparrowDataset
        raster = kwargs.get( 'raster', "False")
        raw_result: xa.DataArray = self._model.predict( data )
        dset: SparrowDataset = self.dataloader.dataset
        species = dset.Y.species.values
        result: xa.DataArray = raw_result.rename(y='species').assign_coords( species=species )
        return self.dataloader.toraster(result) if raster else result

    def feature_attribution(self, data: xa.DataArray) -> List[np.ndarray]:
        return self._model.feature_attribution(data)

    def event(self, source: str, event ):
        print( f"Processing event[{source}]: {event}")

    # def plot_species_prediction(self, title: str, result: xa.DataArray, iS: int, **kwargs ) -> FigureCanvasBase:
    #     from sparrow.occurrences.manager import Species
    #     fig: Figure = plt.figure( title, figsize=kwargs.get('figsize',(14,7)) )
    #     fig.canvas.mpl_connect('button_press_event', partial(self.event,"species_prediction") )
    #     s: Species = self.dataloader.occ.species[iS]
    #     gs = GridSpec(12, 8, figure=fig)
    #     ax: Axes = fig.add_subplot( gs[:8,:4], projection=PlateCarree() )
    #     ax.add_feature(cfeature.COASTLINE, zorder=10, color="magenta" )
    #     result_map: xa.DataArray = self.dataloader.toraster( result[:, iS] )
    #     extent = xextent( result_map )
    #     ax.set_extent(extent)
    #     ax.imshow(result_map.values, cmap="jet", origin="lower", extent=extent, zorder=5)
    #     ax.set_title("Predicted Species Distribution", {'fontsize': 8})
    #
    #     ax: Axes = fig.add_subplot( gs[:8,4:], projection=PlateCarree())
    #     ax.set_extent(extent)
    #     ax.add_feature(cfeature.OCEAN)
    #
    #     pts = s.points(extent[:2],extent[2:])
    #     ax.scatter(pts['x'], pts['y'], s=1, color='red')
    #     ax.set_title("occurrences", {'fontsize': 8})
    #
    #     ax: Axes = fig.add_subplot( gs[8:,:4] )
    #     self.plot_loss_element(ax, iS)
    #     ax.set_title("Training Loss", {'fontsize': 8})
    #
    #     ax: Axes = fig.add_subplot( gs[8:,4:] )
    #     self.plot_loss_metrics(ax, iS)
    #     ax.set_title("Loss Metrics", {'fontsize': 8})
    #
    #     fig.tight_layout()
    #     return fig.canvas

 #    def hvplot_species_prediction(self, title: str, result: xa.DataArray, iS: int, **kwargs ) -> FigureCanvasBase:
 #        from sparrow.occurrences.manager import Species
 #        s: Species = self.dataloader.occ.species[iS]
 #        result_map: xa.DataArray = self.dataloader.toraster( result[:, iS] )
 # #       extent = xextent( result_map )
 #        feature_plot = result_map.hvplot.image( x = 'lon', y = 'lat', cmap = "jet", width=600, height=500 )
 # #       z = 'air', groupby = 'time', cmap = 'kbc_r')
 # #       feature_plot = hv.Image(band).opts(cmap="jet", width=plot_size[0], height=plot_size[1], tools=["hover"], nodata=nodata)
 #
 #
 #        # pts = s.points(extent[:2],extent[2:])
 #        # ax.scatter(pts['x'], pts['y'], s=1, color='red')
 #        # self.plot_loss_element(ax, iS)
 #        # self.plot_loss_metrics(ax, iS)
 #
 #        return feature_plot
