from spectraclass.widgets.points import PointsInteractor
from matplotlib.backend_bases import PickEvent, MouseButton  # , NavigationToolbar2
from spectraclass.data.spatial.tile.tile import Block
import os, logging, numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Type, Iterable
from spectraclass.util.logs import LogManager, lgm, exception_handled

def pid( instance ): return hex(id(instance))[-4:]

class Marker:
    def __init__(self, pids: Union[np.ndarray,Iterable], cid: int, **kwargs ):
        self.cid = cid
        self.args = kwargs
        self._pids: np.ndarray = pids if isinstance( pids, np.ndarray ) else np.array(pids)

    @property
    def pids(self) -> np.ndarray:
        return self._pids

    def isTransient(self):
        return self.cid == 0

    def __eq__(self, m ):
        return isinstance( m, Marker ) and ( m.cid == self.cid ) and ( m.pids.tolist() == self._pids.tolist() )

    def __ne__(self, m ):
        return not self.__eq__( m )

    @property
    def size(self):
        return self._pids.size

    def isEmpty(self):
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

    def __init__(self, ax, block: Block ):
        super(MarkerManager, self).__init__( ax )
        self._block: Block = block
        self._adding_marker = False

    def set_block(self, block ):
        self._block = block

    def delete_marker(self, y, x ):
        from spectraclass.model.labels import LabelsManager, lm
        pindex = self._block.coords2pindex( y, x )
        lgm().log( f" delete_marker: pid = {pindex}" )
        lm().deletePid( pindex )

    def get_points( self ) -> Tuple[ List[float], List[float], List[str] ]:
        from spectraclass.model.labels import LabelsManager, lm
        ycoords, xcoords, colors, markers = [], [], [], lm().markers
        for marker in markers:
            for pid in marker.pids:
                coords = self._block.pindex2coords( pid )
                if (coords is not None) and self._block.inBounds( coords['y'], coords['x'] ):   #  and not ( labeled and (c==0) ):
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
            if marker is None:
                lgm().log("NULL Marker: point select is probably out of bounds.")
            else:
                app().add_marker("map", marker)
        self._adding_marker = False

    @exception_handled
    def on_button_press(self, event):
        from spectraclass.model.labels import LabelsManager, lm
        if (event.xdata != None) and (event.ydata != None) and (event.inaxes == self.ax) and self._enabled:
            pid = self._block.coords2pindex(event.ydata, event.xdata)
            lgm().log(f"\n on_button_press --> selected pid = {pid}, button = {event.button}" )
            if pid >= 0:
                if int(event.button) == self.RIGHT_BUTTON:
                    lm().deletePid( pid )
                else:
                    cid = lm().current_cid
        #           ptindices = self._block.pindex2indices(pid)
         #           lgm().log(f"Adding marker for pid = {pid}, cid = {cid}, ptindices= {ptindices}, coords = {[event.xdata,event.ydata]}")
         #           classification = self.label_map.values[ ptindices['iy'], ptindices['ix'] ] if (self.label_map is not None) else -1
                    self.add( Marker( [pid], cid )) #, classification = classification ) )
                self.plot()


