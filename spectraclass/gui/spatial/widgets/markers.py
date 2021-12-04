from spectraclass.widgets.points import PointsInteractor
from matplotlib.backend_bases import PickEvent, MouseButton  # , NavigationToolbar2
from spectraclass.util.logs import LogManager, lgm, exception_handled
from spectraclass.data.spatial.tile.tile import Block
from typing import List, Union, Tuple, Optional, Dict, Callable
from spectraclass.model.labels import LabelsManager, lm
from spectraclass.model.base import Marker

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
        pindex = self._block.coords2pindex( y, x )
        lm().deletePid( pindex )

    def get_points( self ) -> Tuple[ List[float], List[float], List[str] ]:
        ycoords, xcoords, colors, markers = [], [], [], lm().markers
        lgm().log(f" ** get_markers, #markers = {len(markers)}")
        for marker in markers:
            for pid in marker.pids:
                coords = self._block.pindex2coords( pid )
                if (coords is not None) and self._block.inBounds( coords['y'], coords['x'] ):   #  and not ( labeled and (c==0) ):
                    ycoords.append( coords['y'] )
                    xcoords.append( coords['x'] )
                    colors.append( lm().colors[marker.cid] )
        return ycoords, xcoords, colors

    def on_pick( self, event: PickEvent ):
        rightButton: bool = event.mouseevent.button == MouseButton.RIGHT
        if ( event.name == "pick_event" ) and ( event.artist == self.points ) and rightButton: #  and ( self.key_mode == Qt.Key_Shift ):
            self.delete_marker( event.mouseevent.ydata, event.mouseevent.xdata )
            self.plot()

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
        if event.xdata != None and event.ydata != None:
            inaxes = (event.inaxes == self.ax)
            if inaxes and self._enabled:
                rightButton: bool = int(event.button) == self.RIGHT_BUTTON
                pid = self._block.coords2pindex( event.ydata, event.xdata )
                lgm().log( f" --> selected pid = {pid}" )
                if pid >= 0:
                    cid = lm().current_cid
        #           ptindices = self._block.pindex2indices(pid)
         #           lgm().log(f"Adding marker for pid = {pid}, cid = {cid}, ptindices= {ptindices}, coords = {[event.xdata,event.ydata]}")
         #           classification = self.label_map.values[ ptindices['iy'], ptindices['ix'] ] if (self.label_map is not None) else -1
                    self.add( Marker( [pid], cid )) #, classification = classification ) )
                else:
                    lgm().log(f"Can't add marker, pid = {pid}")