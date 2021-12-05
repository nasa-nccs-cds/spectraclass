from spectraclass.data.base import DataManager
from spectraclass.gui.spatial.application import Spectraclass
from spectraclass.gui.points import PointCloudManager
from spectraclass.model.labels import LabelsManager
from spectraclass.gui.spatial.widgets.markers import Marker
import xarray as xa

app = Spectraclass.instance()
pcm = PointCloudManager.instance()
dm: DataManager = app.configure("demo1",'aviris')
lm = LabelsManager.instance()

lm.setLabels( [ ('Class-1', "cyan"),  ('Class-2', "green"),  ('Class-3', "magenta"),  ('Class-4', "blue")] )
pcm.init_data()
model_data: xa.DataArray = dm.getModelData()
LabelsManager.instance().initLabelsData( model_data )
lm.addMarker( Marker([250],1) )

dm.distance()



