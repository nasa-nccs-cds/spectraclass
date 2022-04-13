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
        from spectraclass.data.spatial.tile.manager import tm
        self.cid = cid
        self.type = type
        self.props = kwargs
        self.block_index = kwargs.get( 'block_index', tm().block_index )
        self.image_index = kwargs.get( 'image_index', tm().image_index )
        self._pids: np.ndarray = pids if isinstance( pids, np.ndarray ) else np.array( pids, dtype=np.int64 )
        self._mask: Optional[np.ndarray] = kwargs.get( 'mask', None )

    @property
    def pids(self) -> np.ndarray:
        return self._pids

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

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
        return isinstance( m, Marker ) and ( m.cid == self.cid ) and ( m.pids == self.pids )

    def __ne__(self, m ):
        return not self.__eq__( m )

    def __str__(self):
        return f"Marker[{self.cid}]: {self.size} pids"

    @property
    def size(self):
        return self._pids.size

    @property
    def empty(self):
        return self._pids.size == 0

    def deletePid( self, pid: int ) -> bool:
        try:
            new_PIDS = self._pids[ self._pids != pid ]
       #     lgm().log( f"  ********>> Marker[{self.cid}].deletePid[{pid}]: {self._pids.tolist()} -> {new_PIDS.tolist()}")
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

    def __init__(self, ax ):
        super(MarkerManager, self).__init__( ax )
        self._adding_marker = False
        self._markers = {}

    @property
    def block(self) -> Block:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return tm().getBlock()

    @exception_handled
    def delete_marker(self, x, y ):
        apx, apy = self.ax.transData.transform( (x, y) )
        for (pid, marker) in self._markers.items():
            (mx,my) = marker.props['point']
            amx, amy = self.ax.transData.transform( (mx, my) )
            if (abs(apx-amx)<self.PICK_DIST) and (abs(apy-amy)<self.PICK_DIST):
                self.remove( pid )
        self.plot()

    @exception_handled
    def get_highlight_points( self ) -> Tuple[ List[float], List[float], List[int] ]:
        ycoords, xcoords, cids = [], [], []
        for (pid,cid) in self._highlight_points:
            coords = self.block.pid2coords(pid)
            if (coords is not None) and self.block.inBounds( coords['y'], coords['x'] ):   #  and not ( labeled and (c==0) ):
                ycoords.append( coords['y'] )
                xcoords.append( coords['x'] )
                cids.append( cid )
        return ( ycoords, xcoords, cids )

    def get_points( self ) -> Tuple[ List[float], List[float], List[str] ]:
        from spectraclass.model.labels import LabelsManager, lm
        ycoords, xcoords, colors, markers = [], [], [], lm().markers
        for marker in markers:
            if marker.type == "marker":
                point = marker['point']
                if point is not None:
               #     lgm().log(f" ** get_points, point = {marker.pids}")
                    ycoords.append(point[1])
                    xcoords.append(point[0])
                    colors.append(lm().colors[marker.cid])
                else:
                    lgm().log( f" ** get_points, markers = {marker.pids}")
                    for pid in marker.pids:
                        coords = self.block.pid2coords(pid)
                        if (coords is not None) and self.block.inBounds( coords['y'], coords['x'] ):   #  and not ( labeled and (c==0) ):
                            ycoords.append( coords['y'] )
                            xcoords.append( coords['x'] )
                            colors.append( lm().colors[marker.cid] )
        return ycoords, xcoords, colors


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
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        marker = self._markers.pop( pid, None )
        if marker is not None:
            lm().deletePid( pid )
            gpm().remove_point( pid )
            pcm().deleteMarkers( [pid] )

    @log_timing
    def mark_point(self, pid, **kwargs ) -> Optional[Tuple[float,float]]:
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        if pid >= 0:
            cid = kwargs.get( 'cid', lm().current_cid )
            point = kwargs.get('point', mm().get_point_coords( pid ) )
            m = Marker( "marker", [pid], cid, point=point )
            self.add(m)
            return point

    @exception_handled
    def on_button_press(self, event: MouseEvent ):
        if (event.xdata != None) and (event.ydata != None) and (event.inaxes == self.ax) and self._enabled:
            if int(event.button) == self.RIGHT_BUTTON:
                self.delete_marker( event.xdata, event.ydata )
            elif int(event.button) == self.LEFT_BUTTON:
                pid = self.block.coords2pindex(event.ydata, event.xdata)
                lgm().log(f"on_button_press --> selected pid = {pid}, button = {event.button}")
                self.mark_point( pid, point=(event.xdata,event.ydata) )
            self.plot()


