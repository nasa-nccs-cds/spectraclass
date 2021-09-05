from typing import List, Dict, Tuple, Optional
import math, atexit, os, traceback
from matplotlib.widgets import PolygonSelector,  RectangleSelector, Button, Slider, _SelectorWidget
from matplotlib.lines import Line2D
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from spectraclass.gui.control import UserFeedbackManager, ufm
from matplotlib.patches import Rectangle
# plt.rcParams['toolbar'] = 'toolmanager'
#
class LassoSelector(_SelectorWidget):

    def __init__(self, ax, onselect=None, useblit=True, lineprops=None,  button=None):
        super().__init__(ax, onselect, useblit=useblit, button=button)
        self.verts = None
        if lineprops is None:
            lineprops = dict()
        lineprops.update(animated=self.useblit, visible=False)
        self.line = Line2D([], [], **lineprops)
        self.ax.add_line(self.line)
        self.artists = [self.line]

    def onpress(self, event):
        self.press(event)

    def _press(self, event):
        self.verts = [self._get_data(event)]
        self.line.set_visible(True)

    def onrelease(self, event):
        self.release(event)

    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.onselect(self.verts)

    def _onmove(self, event):
        if self.verts is None:
            return
        self.verts.append(self._get_data(event))
        self.line.set_data(list(zip(*self.verts)))
        self.update()

class SelectionTool(ToolToggleBase):

    def __init__( self, figure, **kwargs ):
        super().__init__( figure.canvas.manager.toolmanager, self.__class__.__name__, toggled=False, **kwargs )
        self.figure = figure

    def selection(self) -> List[Tuple]:
        raise NotImplementedError()

    def enable(self, *args ):
        raise NotImplementedError()

    def disable(self, *args):
        self.disconnect()
        super().disable(self)

    def onselect(self, *args):
        print(f"onselect: {args} ")
        self.canvas.draw_idle()

    def disconnect(self):
        raise NotImplementedError()


class LassoSelectionTool(SelectionTool):
    default_keymap = 'r'
    description = 'Select region with curve'

    def __init__( self, figure, **kwargs ):
        super().__init__( figure, **kwargs )
        self.curve: LassoSelector = None

    def selection(self) -> List[Tuple]:
        return self.curve.verts

    def enable(self, *args ):
       ufm().show( "Enable Lasso Selection")
       self.curve: LassoSelector = LassoSelector(self.figure.axes[0], self.onselect, useblit=False, button=[1] )

    def disconnect(self):
        print(f"DISCONNECT")
        self.curve.set_visible(False)
        self.curve.set_active(False)
        self.curve = None
        self.canvas.draw_idle()

class RectangleSelectionTool(SelectionTool):
    default_keymap = 'r'
    description = 'Select Rectangular Region'

    def __init__( self, figure, **kwargs ):
        super().__init__( figure, **kwargs )
        self.poly: RectangleSelector = None

    def selection(self) -> List[Tuple]:
        xc, yc = self.poly.corners
        return list( zip( xc, yc ) )

    def enable(self, *args ):
       ufm().show( "Enable Rectangle Selection")
       self.poly = RectangleSelector( self.figure.axes[0], self.onselect, drawtype='box', useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    def disconnect(self):
        print(f"DISCONNECT")
        self.poly.set_visible(False)
        self.poly.set_active( False )
        self.poly = None
        self.canvas.draw_idle()

class PolygonSelectionTool(SelectionTool):
    default_keymap = 'p'
    description = 'Select Polygonal Region'

    def __init__( self, figure, **kwargs ):
        super().__init__( figure, **kwargs )
        self.figure = figure
        self.poly = None

    def selection(self) -> List[Tuple]:
        assert ( (self.poly is not None) and self.poly._polygon_completed ), "Region boundary is not well defined"
        return self.poly.verts

    def enable(self, *args ):
       ufm().show( "Enable Polygon Selection")
       self.poly = PolygonSelector( self.figure.axes[0], self.onselect )

    def disconnect(self):
        print(f"DISCONNECT")
        self.poly.set_visible(False)
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