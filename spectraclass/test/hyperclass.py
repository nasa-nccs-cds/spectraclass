from spectraclass.gui.spatial.application import Spectraclass
from spectraclass.model.labels import LabelsManager

classes = [ ('Class-1', "cyan"),
            ('Class-2', "green"),
            ('Class-3', "magenta"),
            ('Class-4', "blue")]
LabelsManager.instance().setLabels( classes )

app: Spectraclass = Spectraclass.instance()
app.gui( False )