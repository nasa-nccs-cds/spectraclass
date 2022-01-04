import xarray as xa
from spectraclass.widgets.points import PointsInteractor
from matplotlib.backend_bases import PickEvent, MouseButton, MouseEvent  # , NavigationToolbar2
from matplotlib.path import Path
from spectraclass.data.spatial.tile.tile import Block
import os, time, logging, numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type, Iterable
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing

def pid( instance ): return hex(id(instance))[-4:]

class Marker:
    def __init__(self, type: str, pids: Union[np.ndarray,Iterable], cid: int, **kwargs ):
        self.cid = cid
        self.type = type
        self.props = kwargs
        self._pids: np.ndarray = pids if isinstance( pids, np.ndarray ) else np.array(pids)

    @property
    def pids(self) -> np.ndarray:
        return self._pids

    @property
    def colors(self) -> List[str]:
        from spectraclass.model.labels import LabelsManager, lm
        return [ lm().colors[ self.cid ] ] * self._pids.size

    def isTransient(self):
        return self.cid == 0

    def __setitem__( self, key: str, item ):
        self.props[key] = item

    def __getitem__(self, key: str ):
        return self.props.get(key,None)

    def __eq__(self, m ):
        return isinstance( m, Marker ) and ( m.cid == self.cid ) and ( m.pids.tolist() == self._pids.tolist() )

    def __ne__(self, m ):
        return not self.__eq__( m )

    def __str__(self):
        return f"Marker[{self.cid}]: {self._pids.size()} pids"

    @property
    def size(self):
        return self._pids.size

    @property
    def empty(self):
        return self._pids.size == 0

    def deletePid( self, pid: int ) -> bool:
        try:
            new_PIDS = self._pids[ self._pids != pid ]
            lgm().log( f"  ********>> Marker[{self.cid}].deletePid[{pid}]: {self._pids.tolist()} -> {new_PIDS.tolist()}")
            self._pids = new_PIDS
            return True
        except: return False

    def deletePids( self, dpids: np.ndarray ) -> bool:
        try:
            self._pids = np.setdiff1d( self._pids, dpids )
            return True
        except: return False

    def tostr(self):
        return self.__repr__()

    def __repr__(self):
        if self.size > 10:
            return f"Marker[{self.cid}]-{self.size} )"
        else:
            return f"Marker[{self.cid}]: {self._pids.tolist()} )"

class MarkerManager( PointsInteractor ):

    RIGHT_BUTTON = 3
    MIDDLE_BUTTON = 2
    LEFT_BUTTON = 1
    PICK_DIST = 5

    def __init__(self, ax, block: Block ):
        super(MarkerManager, self).__init__( ax )
        self._block: Block = block
        self._adding_marker = False
        self._markers = {}

    def set_block(self, block ):
        self._block = block

    @exception_handled
    def delete_marker(self, x, y ):
        apx, apy = self.ax.transData.transform( (x, y) )
        for (pid, marker) in self._markers.items():
            (mx,my) = marker.props['point']
            amx, amy = self.ax.transData.transform( (mx, my) )
            if (abs(apx-amx)<self.PICK_DIST) and (abs(apy-amy)<self.PICK_DIST):
                self.remove( pid )
        self.plot()

    @log_timing
    def get_points( self ) -> Tuple[ List[float], List[float], List[str] ]:
        from spectraclass.model.labels import LabelsManager, lm
        ycoords, xcoords, colors, markers = [], [], [], lm().markers
        for marker in markers:
            if marker.type == "marker":
                point = marker['point']
                if point is not None:
                    ycoords.append(point[1])
                    xcoords.append(point[0])
                    colors.append(lm().colors[marker.cid])
                else:
                    for pid in marker.pids:
                        coords = self._block.pindex2coords( pid )
                        if (coords is not None) and self._block.inBounds( coords['y'], coords['x'] ):   #  and not ( labeled and (c==0) ):
                            ycoords.append( coords['y'] )
                            xcoords.append( coords['x'] )
                            colors.append( lm().colors[marker.cid] )
        return ycoords, xcoords, colors

    @log_timing
    def get_classmap(self) -> xa.DataArray:
        from spectraclass.model.labels import LabelsManager, lm
        cmap: xa.DataArray = self._block.classmap()
        for marker in lm().markers:
            if marker.type == "label":
                for pid in marker.pids:
                    idx = self._block.pindex2indices( pid )
                    cmap[ idx['iy'], idx['ix'] ] = marker.cid
        return cmap


    # def on_pick( self, event: PickEvent ):
    #     rightButton: bool = event.mouseevent.button == MouseButton.RIGHT
    #     lgm().log(f"\n ** on_pick, rightButton = {rightButton}, inpoints = {( event.artist == self.points )}")
    #     if ( event.name == "pick_event" ) and ( event.artist == self.points ) and rightButton: #  and ( self.key_mode == Qt.Key_Shift ):
    #         self.delete_marker( event.mouseevent.ydata, event.mouseevent.xdata )
    #         self.plot()

    @exception_handled
    def add( self, marker: Marker ):
        from spectraclass.application.controller import app
        if not self._adding_marker:
            self._adding_marker = True
            if (marker is None) or (len(marker.pids) == 0):
                lgm().log("NULL Marker: point select is probably out of bounds.")
            else:
                app().add_marker("map", marker)
                self._markers[ marker.pids[0] ] = marker
        self._adding_marker = False

    def remove( self, pid: int ):
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        marker = self._markers.pop( pid, None )
        if marker is not None:
            lm().deletePid( pid )
            gpm().remove_point( pid )

    @exception_handled
    def on_button_press(self, event: MouseEvent ):
        from spectraclass.model.labels import LabelsManager, lm
        if (event.xdata != None) and (event.ydata != None) and (event.inaxes == self.ax) and self._enabled:
            if int(event.button) == self.RIGHT_BUTTON:
                self.delete_marker( event.xdata, event.ydata )
            elif int(event.button) == self.LEFT_BUTTON:
                pid = self._block.coords2pindex(event.ydata, event.xdata)
                lgm().log(f"\n on_button_press --> selected pid = {pid}, button = {event.button}")
                if pid >= 0:
                    m = Marker( "marker", [pid], lm().current_cid, point=(event.xdata,event.ydata) )
                    self.add( m )
            self.plot()


