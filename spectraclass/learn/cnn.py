from spectraclass.data.spatial.tile.tile import Block
import tensorflow as tf, xarray as xa
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from typing import List, Tuple, Optional, Dict
from spectraclass.learn.models.network import Network
from spectraclass.learn.base import ModelType

def osize( isize: int, ks: int, s: int  ) -> int:
    return ((isize-ks)//s) + 1

class CNN(Network):
    TYPE = ModelType.SPATIAL

    def _build_model(self, **kwargs) -> Tuple[tf.keras.models.Model,Dict]:
        from spectraclass.learn.models.spatial import SpatialModelWrapper
        nfeatures = kwargs.pop('nfeatures', 32 )
        from spectraclass.model.labels import lm
        input_shape = SpatialModelWrapper.get_input_shape()
        nclasses = lm().nLabels
        ks = kwargs.pop('kernel_size',3)
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.Input( shape=input_shape ) )
        model.add( tf.keras.layers.Conv2D( nfeatures, (ks,ks), activation='relu', padding="same" ) )
        model.add( tf.keras.layers.Reshape( SpatialModelWrapper.flatten(input_shape,nfeatures) ) )
        model.add( tf.keras.layers.Dense( 2*nfeatures, activation='relu' ) )
        model.add( tf.keras.layers.Dense( nfeatures, activation='relu') )
        model.add( tf.keras.layers.Dense( nclasses, activation='softmax' ) )
        return model, kwargs

class SpectralCNN(Network):
    TYPE = ModelType.SPECTRAL

    def _build_model(self, **kwargs) -> Tuple[tf.keras.models.Model,Dict]:
        from spectraclass.data.spatial.tile.manager import tm
        from spectraclass.model.labels import lm
        pd: xa.DataArray = tm().getBlock().getPointData()[0]
        input_shape: List[int] = list(pd.expand_dims("channels",2).shape[1:])
        nb, nfeatures = input_shape[0], -1
        nclasses = lm().nLabels
        cnn_layers = kwargs.pop('cnn_layers', [(8,5,3),(8,5,3),(8,5,3)] )
        dense_layers = kwargs.pop('dense_layers', [32, 16])
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.Input( shape=input_shape ) )
        for (nf,ks,s) in cnn_layers:
            model.add( tf.keras.layers.Conv1D(filters=nf, kernel_size=ks, activation='relu', padding="valid", strides=s ) )
            nb, nfeatures = osize( nb, ks, s ), nf
        oshape = [ nb*nfeatures ]
        lgm().log( f"Reshape: input={input_shape}, model_output={model.output_shape}, nb={nb}, nf={nfeatures}, oshape={oshape}")
        model.add( tf.keras.layers.Reshape( oshape ) )
        for nf in dense_layers:
            model.add( tf.keras.layers.Dense( nf, activation='relu' ) )
        model.add( tf.keras.layers.Dense( nclasses, activation='softmax' ) )
        return model, kwargs

class CNN3D(Network):
    TYPE = ModelType.SPECTRALSPATIAL

    def _build_model(self, **kwargs) -> Tuple[tf.keras.models.Model,Dict]:
        from spectraclass.data.spatial.tile.manager import tm
        from spectraclass.model.labels import lm
        pd: xa.DataArray = tm().getBlock().getPointData()[0]
        input_shape: List[int] = list(pd.expand_dims("channels",2).shape[1:])
        nb, nfeatures = input_shape[0], -1
        nclasses = lm().nLabels
        cnn_layers = kwargs.pop('cnn_layers', [(5,5,3),(4,5,3),(3,5,3)] )
        dense_layers = kwargs.pop('dense_layers', [32, 16])
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.Input( shape=input_shape ) )
        for (nf,ks,s) in cnn_layers:
            model.add( tf.keras.layers.Conv3D(filters=nf, kernel_size=(ks,3,3), activation='relu', padding="valid", strides=(s,1,1) ) )
            nb, nfeatures =  nb//s, nf
        oshape = [ nb*nfeatures ]
        lgm().log( f"Reshape: input={input_shape}, model_output={model.output_shape}, nb={nb}, nf={nfeatures}, oshape={oshape}")
        model.add( tf.keras.layers.Reshape( oshape ) )
        for nf in dense_layers:
            model.add( tf.keras.layers.Dense( nf, activation='relu' ) )
        model.add( tf.keras.layers.Dense( nclasses, activation='softmax' ) )
        return model, kwargs
# block: Block = tm().getBlock()
# point_data: xa.DataArray = block.getPointData()[0].expand_dims("channels", 2)
#


