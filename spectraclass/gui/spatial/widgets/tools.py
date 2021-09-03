from collections import OrderedDict
from functools import partial
import traitlets as tl
import ipywidgets as ipw
from spectraclass.util.logs import LogManager, lgm, exception_handled
import types, pandas as pd
import xarray as xa
import numpy as np
from typing import List, Dict, Tuple, Optional
import math, atexit, os, traceback
from matplotlib.widgets import PolygonSelector
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from  ipympl.backend_nbagg import Toolbar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from matplotlib.figure import Figure, FigureCanvasBase
from matplotlib.image import AxesImage
from matplotlib.backend_bases import PickEvent, MouseButton  # , NavigationToolbar2
from spectraclass.gui.control import UserFeedbackManager, ufm
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
# plt.rcParams['toolbar'] = 'toolmanager'

class PolygonSelectionTool(ToolToggleBase):
    default_keymap = 'p'
    description = 'Select Polygonal Region'

    def __init__( self, figure, **kwargs ):
        super().__init__( figure.canvas.manager.toolmanager, "PolygonSelectionTool", toggled=False, **kwargs )
        self.figure = figure
        self.poly = None

    def selection(self) -> List[Tuple]:
        assert ( (self.poly is not None) and self.poly._polygon_completed ), "Region boundary is not well defined"
        return self.poly.verts

    def enable(self, event=None):
       ufm().show( "Enable PolygonSelection")
       self.poly = PolygonSelector( self.figure.axes[0], self.onselect )

    def disable(self, event=None):
        self.poly.set_visible(False)
        self.disconnect()
        super().disable( self, event=None )

    def onselect(self, *args ):
        print( f"onselect: {args} ")
        self.canvas.draw_idle()

    def disconnect(self):
        print(f"DISCONNECT")
        self.poly.disconnect_events()
        self.poly = None
        self.canvas.draw_idle()

class PageSlider(Slider):

    def __init__(self, ax: Axes, numpages = 10, valinit=0, valfmt='%1d', **kwargs ):
        self.facecolor=kwargs.get('facecolor',"yellow")
        self.activecolor = kwargs.pop('activecolor',"blue" )
        self.stepcolor = kwargs.pop('stepcolor', "#ff6f6f" )
        self.on_animcolor = kwargs.pop('on-animcolor', "#006622")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.maxIndexedPages = 24
        self.numpages = numpages
        self.axes = ax

        super(PageSlider, self).__init__(ax, "", 0, numpages, valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        indexMod = math.ceil( self.numpages / self.maxIndexedPages )
        for i in range(numpages):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = Rectangle((float(i)/numpages, 0), 1./numpages, 1, transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            if i % indexMod == 0:
                ax.text(float(i)/numpages+0.5/numpages, 0.5, str(i+1), ha="center", va="center", transform=ax.transAxes, fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = Button(bax, label='$\u25C1$', color=self.stepcolor, hovercolor=self.activecolor)
        self.button_forward = Button(fax, label='$\u25B7$', color=self.stepcolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self. forward)

    def refesh(self):
        self.axes.figure.canvas.draw()

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >=self.valmax: return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event=None):
        current_i = int(self.val)
        i = current_i+1
        if i >= self.valmax: i = self.valmin
        self.set_val(i)
        self._colorize(i)

    def backward(self, event=None):
        current_i = int(self.val)
        i = current_i-1
        if i < self.valmin: i = self.valmax -1
        self.set_val(i)
        self._colorize(i)