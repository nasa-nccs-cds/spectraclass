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
        self._pids: np.ndarray = pids if isinstance( pids, np.ndarray ) else np.array( list(pids), dtype=np.int64 )
        self._mask: Optional[np.ndarray] = kwargs.get( 'mask', None )

    def bid(self) -> Tuple[int,Tuple]:
        return (  self.image_index, self.block_coords )

    @property
    def oid(self) -> str:
        return hex(id(self))

    @property
    def block_coords(self) -> Tuple:
        return tuple(self.block_index)

    def active(self, **kwargs) -> bool:
        from spectraclass.data.spatial.tile.manager import tm
        block: Block = kwargs.get( 'block', tm().getBlock() )
        rv = ( self.block_coords == block.block_coords ) and (self.image_index == block.tile_index )
        if not rv: lgm().log( f"Rejecting inactive marker: {self.block_coords} <-> {block.block_coords}" )
        return rv

    def relevant(self, mtype: str, **kwargs ) -> bool:
        return self.active(**kwargs) and ((mtype is None) or (self.type == mtype))

    @property
    def gids(self) -> np.ndarray:
        return self._pids

    def set_pids(self, pids: np.ndarray):
        self._pids = pids

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
        rv = isinstance( m, Marker ) and ( m.cid == self.cid ) and (m.gids == self.gids)
        return rv.all() if isinstance( rv, np.ndarray ) else rv

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
        self._adding_marker = False
        self._markers = {}
        self._probes = {}
        super(MarkerManager, self).__init__( ax )

    def clear(self):
        self._markers = {}
        self._probes = {}

    @property
    def block(self) -> Block:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return tm().getBlock()

    @exception_handled
    def delete_marker(self, x, y ):
        apx, apy = self.ax.transData.transform( (x, y) )
        for pid in list(self._markers.keys()):
            marker = self._markers[pid]
            (mx,my) = marker.props['point']
            amx, amy = self.ax.transData.transform( (mx, my) )
            if (abs(apx-amx)<self.PICK_DIST) and (abs(apy-amy)<self.PICK_DIST):
                self.remove( pid )
        self.plot()

    @exception_handled
    def get_highlight_points( self ) -> Tuple[ List[float], List[float], List[int] ]:
        ycoords, xcoords, cids = [], [], []
        for (pid,cid) in self._highlight_points:
            coords = self.block.gid2coords(pid)
            if (coords is not None) and self.block.inBounds( coords['y'], coords['x'] ):   #  and not ( labeled and (c==0) ):
                ycoords.append( coords['y'] )
                xcoords.append( coords['x'] )
                cids.append( cid )
        return ( ycoords, xcoords, cids )

    def get_points( self, probes: bool = False  ) -> Tuple[ List[float], List[float], List[str] ]:
        from spectraclass.model.labels import LabelsManager, lm
        ycoords, xcoords, colors = [], [], []
        lgm().log(f" *** get_points(probe={probes}): #markers = {len(self._markers)}" )
        for marker in self._markers.values():
            valid =  (marker.cid == 0) if probes else (marker.cid > 0)
            lgm().log(f" *** >>> Marker[{marker.cid}] valid={valid} pids={marker.gids} point={marker['point']}")
            if (marker.type == "marker") and valid:
                point = marker['point']
                if point is not None:
                    lgm().log(f" *** >>> point = {marker.gids}")
                    ycoords.append(point[1])
                    xcoords.append(point[0])
                    colors.append(lm().colors[marker.cid])
                else:
                    lgm().log( f" *** >>> markers = {marker.gids}")
                    for pid in marker.gids:
                        coords = self.block.gid2coords(pid)
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

    def _add_marker(self, marker: Marker ):
        for mid in list(self._markers.keys()):
            if self._markers[mid].cid == 0:
                self._markers.pop( mid )
        self._markers[marker.gids[0]] = marker

    @exception_handled
    def add( self, marker: Marker ):
        from spectraclass.application.controller import app
        if not self._adding_marker:
            self._adding_marker = True
            if (marker is None) or (len(marker.gids) == 0):
                lgm().log("NULL Marker: point select is probably out of bounds.")
            else:
                app().add_marker("map", marker)
                self._add_marker( marker )
                self.plot()
        else:
            lgm().log("Dropping marker add: already adding_marker.")
        self._adding_marker = False

    def remove( self, pid: int ):
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.pointcloud import PointCloudManager, pcm
        marker = self._markers.pop( pid, None )
        if marker is not None:
            lm().deletePid( pid )
            gpm().remove_points( [pid], plot=True )
            pcm().deleteMarkers( [pid], plot=True )
        self.plot()

    @log_timing
    def mark_point(self, gid, **kwargs ) -> Optional[Tuple[float,float]]:
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.spatial.map import MapManager, mm
        from spectraclass.data.spatial.tile.manager import tm
        if gid >= 0:
            block = tm().getBlock()
            cid = kwargs.get( 'cid', lm().current_cid )
            point = kwargs.get('point', mm().get_point_coords( gid ) )
            lgm().log( f"mark_point[{cid}], gid={gid}, point={point}")
            m = Marker( "marker", [gid], cid, point=point )
            self.remove( gid )
            self.add(m)
            return point

    @exception_handled
    def on_button_press(self, event: MouseEvent ):
        from spectraclass.gui.spatial.map import MapManager, mm
        from matplotlib.image import AxesImage
        inaxes = event.inaxes == self.ax
        lgm().log(f"MarkerManager.on_button_press --> inaxes = {inaxes}, enabled = {self._enabled}")
        from spectraclass.gui.control import ufm
        if (event.xdata != None) and (event.ydata != None) and inaxes and self._enabled:
            mdata = ""
            if int(event.button) == self.RIGHT_BUTTON:
                self.delete_marker( event.xdata, event.ydata )
            elif int(event.button) == self.LEFT_BUTTON:
                labels_img: AxesImage = mm().layer_managers("labels")[0]
                gid,ix,iy = self.block.coords2gid(event.ydata, event.xdata)
                if labels_img.get_visible():
                    cid = 0 if mm().classification_data is None else mm().classification_data[iy,ix]
                    mdata = mdata + f" label={cid} "
                lgm().log(f" *** --> selected gid = {gid}, button = {event.button}")
                ufm().show( f" event[{event.xdata:.2f},{event.ydata:.2f}]: ({ix},{iy},{gid}) {mdata}" )
                self.mark_point( gid, point=(event.xdata,event.ydata) )
            self.plot()


