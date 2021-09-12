import numpy as np
import sys
from copy import deepcopy
from typing import List, Union, Tuple, Optional, Dict, Callable
from matplotlib.backend_bases import MouseEvent
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class TileSelector:
    INIT_POS = ( sys.float_info.max, sys.float_info.max )

    def __init__( self, ax: Axes, rsize: Tuple[float,float], on_click: Callable[[MouseEvent],None] ):
        self._rsize = rsize
        self._ax = ax
        self.rect = self._rect( self.INIT_POS )
        self._on_click = on_click
        self._background = None
        self._selection_background = None
        self._selection_rect = None
        self._active = False
        self.canvas = None

    def _rect( self, pos: Tuple[float,float] ):
        rect = Rectangle( pos, self._rsize[0], self._rsize[1], facecolor="white", edgecolor="black", alpha=1.0 )
        self._ax.add_patch(rect)
        return rect

    def activate(self):
        self._active = True
        self.canvas = self.rect.figure.canvas
        self.cidpress   = self.canvas.mpl_connect( 'button_press_event', self.on_mouse_click )
        self.cidmotion  = self.canvas.mpl_connect( 'motion_notify_event', self.on_motion )
        self.rect.set_animated(True)
        self.rect.set_visible(True)

    def on_mouse_click( self, event: MouseEvent ):
        if self._selection_background is not None:
            self.canvas.restore_region(self._selection_background )
        self._selection_background = self.canvas.copy_from_bbox( self.rect.axes.bbox )
        if self._selection_rect is None:
            self._selection_rect = self._rect( (event.xdata,event.ydata) )
        else:
            self._selection_rect.set_x( event.xdata )
            self._selection_rect.set_y( event.ydata )
        self._ax.draw_artist(self.rect)
        self._on_click( event )

    def on_motion(self, event: MouseEvent ):
        if (event.inaxes == self.rect.axes):
            self.rect.set_x( event.xdata )
            self.rect.set_y( event.ydata )
            if self._background is not None:
                self.canvas.restore_region(self._background)
            self._background = self.canvas.copy_from_bbox(self.rect.axes.bbox)
            self._ax.draw_artist(self.rect)
            if self._selection_rect is not None:
                self._ax.draw_artist( self._selection_rect )
            self.canvas.blit(self._ax.bbox)
        else:
            if self._background is not None:
                self.canvas.restore_region(self._background)
                self._background = None
                self.canvas.blit(self._ax.bbox)

    def deactivate(self):
        self._active = False
        if self._selection_rect is not None:
            self._selection_rect.set.set_visible( False )
            self._selection_rect = None
        if self._background is not None:
            self.canvas.restore_region( self._background )
            self._background = None
        if self._selection_background is not None:
            self.canvas.restore_region( self._selection_background )
            self._selection_background = None
        self.rect.set_x( self.INIT_POS[0] )
        self.rect.set_y( self.INIT_POS[1] )
        self.canvas.mpl_disconnect( self.cidpress )
        self.canvas.mpl_disconnect( self.cidmotion )
        self.rect.set_animated( False )
        self.rect.set_visible( False )
        self.canvas.draw()
        self.canvas = None

if __name__ == '__main__':

    def on_selection( event: MouseEvent ):
        print( f"Location Selection: {event}")

    fig: plt.Figure = plt.figure()
    ax = fig.add_subplot( 111 )
    ax.set_xlim( -100.0, 100.0 )
    ax.set_ylim( -100.0, 100.0 )
    blocks_per_tile = 5
    dr = TileSelector( ax, blocks_per_tile, on_selection )
    dr.activate()
    plt.show()