import param
import xarray as xa
import hvplot.xarray
import holoviews as hv

class SpectralLayer(param.Parameterized):
    band = param.Integer(default=0)
    alpha = param.Magnitude()
    cmap = param.Selector( objects=hv.plotting.util.list_cmaps(), default="jet" )

    def __init__(self, raster: xa.DataArray, **kwargs ):
        param.Parameterized.__init__(self)
        self._raster = raster
        self._plot_args = kwargs

    def filter_cmaps(self, provider=None, records=False, name=None, category=None, source=None, bg=None, reverse=None ):
        self.cmap.objects = hv.plotting.util.list_cmaps( provider, records, name, category, source, bg, reverse )

    @property
    def layer(self) -> xa.DataArray:
        return self._raster[ self.band ]

    @param.depends('band', 'alpha')
    def plot(self):
        return self.layer.hvplot.image( cmap=self.cmap, **self._plot_args )