from collections import OrderedDict
from typing import List, Union, Dict, Callable, Tuple, Optional, Any
import traitlets.config as tlc
import os, ipywidgets as ipw
from spectraclass.model.base import SCConfigurable
import collections.abc
from functools import partial
import xarray as xa
import numpy as np

def h2c( hexColor: str ) -> List[float]:
    hc = hexColor.strip( "# ")
    cv = [ int(hc[i0:i0+2],16) for i0 in range(0,len(hc),2) ]
    cv = cv if len(cv) == 4 else cv + [255]
    return [ c/255 for c in cv ]

def isIntRGB( color ):
    if isinstance(color, collections.abc.Sequence):
        for val in color:
            if val > 1: return True
    return False

def format_color( color: Union[str,List[Union[float,int]]] ) -> List[float]:
    if isinstance(color, str):  return h2c(color)
    elif isIntRGB(color):       return [c / 255 for c in color]
    else:                       return color

def format_colors( classes: List[Tuple[str,Union[str,List[Union[float,int]]]]] ) -> List[List[float]]:
    return [ format_color(color) for (label, color) in classes ]

def set_alphas( colors, alpha ):
    return [ set_alpha(color, alpha) for color in colors ]

def set_alpha( color, alpha ):
    return color[:3] + [alpha]

class Marker:
    def __init__(self, color: List[float], pids: List[int], cid: int ):
        self.color = color
        self.cid = cid
        self.pids = set(pids)

    def isTransient(self):
        return self.cid == 0

    def isEmpty(self):
        return len( self.pids ) == 0

    def deletePid( self, pid: int ) -> bool:
        try:
            self.pids.remove( pid )
            return True
        except: return False

    def deletePids( self, pids: List[int] ) -> bool:
        try:
            self.pids -= set( pids )
            return True
        except: return False

class Action:
    def __init__(self, type: str, source: str, pids: List[int], cid, **kwargs ):
        self.args = kwargs
        self.type = type
        self.cid=cid
        self.source = source
        self.pids = pids

    def __repr__(self):
        return f"A[{self.type}:{self.source} cid:{self.cid} pids:{self.pids}]"

    def __eq__(self, action: "Action" ):
        return ( self.type  == action.type ) and ( self.cid  == action.cid ) and ( self.source  == action.source ) and ( self.pids  == action.pids )

    @property
    def spec(self):
        return dict( atype=self.type, source=self.source , pids=self.pids, cid=self.cid, **self.args )

class LabelsManager(tlc.SingletonConfigurable, SCConfigurable):
    from ..graph.base import ActivationFlow

    def __init__( self ):
        from ..graph.base import ActivationFlow
        super(LabelsManager, self).__init__()
        self._colors = None
        self._labels = None
        self.buttons = {}
        self.selectedClass = 0
        self._markers: List[Marker] = []
        self._flow: ActivationFlow = None
        self._actions = []
        self._labels_data: xa.DataArray = None
        self._optype = None
        self.template = None
        self.n_spread_iters = 1

    def flow(self) -> Optional[ActivationFlow]:
        return self._flow

    def addAction(self, type: str, source: str, pids: List[int], cid=None, **kwargs ):
        if cid == None: cid = self.selectedClass
        new_action = Action(type, source, pids, cid, **kwargs)
        print(f"ADD ACTION: {new_action}")
        if (len(self._actions) == 0) or (new_action != self._actions[-1]):
            self._actions.append( new_action )

    def popAction(self) -> Optional[Action]:
        try:
            action =  self._actions.pop()
            print( f"POP ACTION: {action}" )
            return action
        except:
            return None

    def undo(self):
        action = self.popAction()
        if action is not None:
            self.processAction( action )

    def processAction(self, action: Action ):
        remaining_markers = []
        for marker in self._markers:
            marker.deletePids( action.pids )
            if not marker.isEmpty(): remaining_markers.append( marker )
        self._markers = remaining_markers

    @property
    def classification(self) -> np.ndarray:
        return self._flow.C

    def initLabelsData( self, point_data: xa.DataArray = None ):
        nodata_value = -1
        if point_data is not None:
            self.template = point_data[:,0].squeeze( drop=True )
            self.template.attrs = point_data.attrs
        if self.template is not None:
            self._labels_data: xa.DataArray = xa.full_like( self.template, 0, dtype=np.int32 ).where( self.template.notnull(), nodata_value )
            self._labels_data.attrs['_FillValue'] = nodata_value
            self._labels_data.name = self.template.attrs['dsid'] + "_labels"
            self._labels_data.attrs[ 'long_name' ] = [ "labels" ]

    def initData( self ):
        from ..graph.manager import ActivationFlowManager
        from spectraclass.data.base import DataManager
        project_data: xa.Dataset = DataManager.instance().loadCurrentProject("graph")
        point_data: xa.DataArray = project_data["plot-y"]
        self.initLabelsData( point_data )
        self._flow = ActivationFlowManager.instance().getActivationFlow( point_data )

    def getMarker( self, pid: int ) -> Optional[Marker]:
        for marker in self._markers:
            if pid in marker.pids: return marker
        return None

    def log_markers(self, msg: str ):
        log_strs = []
        for m in self.markers:
            log_strs.append( f"[{m.cid}:{m.pids[0]}]" if m.size == 1 else  f"M{m.cid}-{m.size}" )
        lgm().log( f"  ----------------------------> log_markers[{msg}]: {' '.join(log_strs)}")

    def updateLabels(self):
        for marker in self._markers:
            for pid in marker.pids:
                self._labels_data[ pid ] = marker.cid

    def labels_data( self, xa = False ) -> Union[xa.DataArray,np.ndarray,None]:
        self.updateLabels()
        result = None
        if self._labels_data is not None:
            result = self._labels_data.copy( self._optype == "distance" )
            if not xa: result = result.values
        return result

    @classmethod
    def getSortedLabels(self, labels_dset: xa.Dataset ) -> Tuple[np.ndarray,np.ndarray]:
        labels: np.ndarray = labels_dset['C'].values
        distance: np.ndarray = labels_dset['D'].values
        indices = np.arange(labels.shape[0])
        indexed_labels = np.vstack( [ indices, labels ] ).transpose()
        selection = (labels > 0)
        filtered_labels = indexed_labels[selection]
        filtered_distance = distance[selection]
        return filtered_labels, filtered_distance

    def spread(self, optype: str,  n_iters = None ) -> Optional[xa.Dataset]:
        if self._flow is None:
            return None
        resume = ( optype == "neighbors" ) and ( self._optype == "neighbors" )
        if not resume: self._flow.clear()
        self._optype = optype
        labels_data = self.labels_data(True)
        niters = self.n_spread_iters if n_iters is None else n_iters
        return self._flow.spread( labels_data, niters )

    def clearTransient(self):
        if len(self._markers) > 0 and self._markers[-1].cid == 0:
            self._markers.pop(-1)

    def clearMarkers(self):
        self._markers = []
        self.initLabelsData()

    def addMarker(self, marker: Marker ):
        self.clearTransient()
        for pid in [ *marker.pids ]: self.deletePid( pid )
        self._markers = list(filter( lambda m: not m.isEmpty(),  self._markers ))
        self._markers.append(marker)

    def popMarker(self) -> Marker:
        marker = self._markers.pop( -1 ) if len( self._markers ) else None
        return marker

    def deletePid(self, pid: int ) -> List[Marker]:
        markers = []
        for marker in self._markers:
            if marker.deletePid( pid ): markers.append( marker )
        return markers

    @property
    def currentMarker(self) -> Marker:
        marker = self._markers[ -1 ] if len( self._markers ) else None
        return marker

    def getMarkers( self ) -> List[Marker]:
        return self._markers

    @property
    def selectedLabel(self):
        return self._labels[ self.selectedClass ]

    def selectedColor(self, mark: bool ) -> Tuple[int,List[float]]:
        icolor = self.selectedClass if mark else 0
        return icolor, self._colors[ icolor ]

    @property
    def colors(self):
        return self._colors

    @property
    def labels(self):
        return self._labels

    @property
    def nLabels(self):
        return len(self._labels)

    def setLabels(self, labels: List[Tuple[str, List[float]]], **kwargs):
        unlabeled_color = kwargs.get( 'unlabeled', [1.0, 1.0, 0.0, 1.0] )
        label_list = [ ('Unlabeled', unlabeled_color ) ] + labels
        self._colors = format_colors( label_list )
        self._labels = [ item[0] for item in label_list ]

    def toDict( self, alpha ) -> OrderedDict:
        labels_dict = OrderedDict()
        for index, label in enumerate(self._labels):
            labels_dict[ label ] = set_alpha( self._colors[index], alpha )
        return labels_dict

    def gui(self, **kwargs ):
        from hyperclass.learn.manager import learningManager
        self.show_unlabeled = kwargs.get( 'show_unlabeled', True )
        with_learning = kwargs.get( 'learning', False )
        self.console = QWidget()
        console_layout = QVBoxLayout()
        self.console.setLayout( console_layout )
        radio_button_style = [ "border-style: outset", "border-width: 4px", "padding: 6px", "border-radius: 10px" ]

        labels_frame = QFrame( self.console )
        buttons_frame_layout = QVBoxLayout()
        labels_frame.setLayout( buttons_frame_layout )
        labels_frame.setFrameStyle( QFrame.StyledPanel | QFrame.Raised )
        labels_frame.setLineWidth( 3 )
        console_layout.addWidget( labels_frame )
        title = QLabel( "Classes" )
        title.setStyleSheet("font-weight: bold; color: black; font: 16pt" )
        buttons_frame_layout.addWidget( title )

        for index, label in enumerate(self._labels):
            raw_color = [str(int(c * 155.99)) for c in self._colors[index]]
            qcolor = [str(150 + int(c * 105.99)) for c in self._colors[index]]
            style_sheet = ";".join( radio_button_style + [f"background-color:rgb({','.join(qcolor)})",f"border-color: rgb({','.join(raw_color)})"] )
            button = ipw.Button( description=label, border= '1px solid gray' )
            button.layout = ipw.Layout( width='auto', flex="1 0 auto" )
            button.on_click( partial( self.onClicked, bindex = index ) )
            self.buttons[ index ] = button

        buttonBox =  ipw.HBox( list(self.buttons.values()) )

        # for index, label in enumerate(self._labels):
        #     if (index > 0) or self.show_unlabeled:
        #         radiobutton = QRadioButton( label, self.console )
        #         radiobutton.index = index
        #         raw_color = [str(int(c * 155.99)) for c in self._colors[index]]
        #         qcolor = [ str(150+int(c*105.99)) for c in self._colors[index] ]
        #         style_sheet = ";".join( radio_button_style + [ f"background-color:rgb({','.join(qcolor)})", f"border-color: rgb({','.join(raw_color)})" ] )
        #         radiobutton.setStyleSheet( style_sheet )
        #         radiobutton.toggled.connect(self.onClicked)
        #         buttons_frame_layout.addWidget( radiobutton )
        #         self.buttons[label] = radiobutton

        console_layout.addStretch( 1 )
        self.buttons[0].setChecked( True )
        return self.console

    def onClicked(self, *args, **kwargs  ):
        selected_index: int = kwargs.get( 'bindex', -1 )
        radioButton = self.buttons[ selected_index ]
        if radioButton.isChecked():
            self.selectedClass = selected_index
            print(f"Selected class {radioButton.index}")

    def setClassIndex(self, cid: int ):
        self.selectedClass = cid
        for button in self.buttons:
            button.setChecked( cid == button.index )
        self.console.update()

labelsManager = LabelsManager()
