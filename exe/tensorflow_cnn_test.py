nfeatures = 64
import tensorflow as tf
from spectraclass.model.labels import lm
from spectraclass.learn.models.spatial import SpatialModelWrapper
from spectraclass.data.base import DataManager

dm: DataManager = DataManager.initialize( "img_mgr", 'aviris' )
location = "desktop"
if location == "adapt":
    dm.modal.cache_dir = "/adapt/nobackup/projects/ilab/cache"
    dm.modal.data_dir = "/css/above/daac.ornl.gov/daacdata/above/ABoVE_Airborne_AVIRIS_NG/data/"
elif location == "desktop":
    dm.modal.cache_dir = "/Volumes/Shared/Cache"
    dm.modal.data_dir = "/Users/tpmaxwel/Development/Data/Aviris"
else: raise Exception( f"Unknown location: {location}")

input_shape = SpatialModelWrapper.get_input_shape()
nclasses = lm().nLabels
ks =  3

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=input_shape))
model.add(tf.keras.layers.Conv2D(nfeatures, (ks, ks), activation='relu', padding="same"))
model.add(tf.keras.layers.Reshape(SpatialModelWrapper.flatten(input_shape, nfeatures)))
model.add(tf.keras.layers.Dense(nfeatures, activation='relu'))
model.add(tf.keras.layers.Dense(nclasses, activation='softmax'))
