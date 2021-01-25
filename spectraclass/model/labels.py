from collections import OrderedDict
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Set
import collections.abc
from functools import partial
import ipywidgets as ipw
import matplotlib.colors as mcolors
from ..graph.manager import ActivationFlow
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm, exception_handled
from spectraclass.model.base import SCSingletonConfigurable, Marker
import xarray as xa
import numpy as np


def c2rgb( color: Union[str,List] ) -> List:
    if isinstance(color, str):  return mcolors.to_rgb(color)
    else:                       return color[:3]

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
    elif isIntRGB(color):       return [c / 255.0 for c in color]
    else:                       return color

def format_colors( classes: List[Tuple[str,Union[str,List[Union[float,int]]]]] ) -> List[List[float]]:
    return [ format_color(color) for (label, color) in classes ]

def set_alphas( colors, alpha ):
    return [ set_alpha(color, alpha) for color in colors ]

def set_alpha( color, alpha ):
    return color[:3] + [alpha]

class Action:
    def __init__(self, type: str, source: str, **kwargs ):
        self.args = kwargs
        self.type = type
        self.source = source

    def __repr__(self):
        return f"A[{self.type}:{self.source} {self.spec}]"

    def __eq__(self, action: "Action" ):
        return ( self.type  == action.type ) and ( self.source  == action.source ) and ( self.spec  == action.spec )

    def __getitem__(self, key: str ):
        return self.args.get( key, None )

    @property
    def spec(self):
        return dict( atype=self.type, source=self.source, **self.args )

def lm() -> "LabelsManager":
    return LabelsManager.instance()

class LabelsManager(SCSingletonConfigurable):

    def __init__(self):
        super(LabelsManager, self).__init__()
        self._colors = None
        self._labels = None
        self._flow: ActivationFlow = None
        self._actions: List[Action] = []
        self._labels_data: xa.DataArray = None
        self._selected_class = 0
        self._nodata_value = -1
        self._optype = None
        self.template = None
        self.n_spread_iters = 1
        self.wSelectedClass: ipw.HBox = None
        self._buttons = []

    def clear_pids(self, cid: int, pids: np.ndarray, **kwargs):
        pass

    @property
    def current_class(self) -> str:
        return self._labels[ self._selected_class ]

    @property
    def current_cid(self) -> int:
        return self._selected_class

    def set_selected_class(self, iclass, *args ):
        from spectraclass.gui.control import UserFeedbackManager, ufm
        ufm().clear()
        self._selected_class = iclass
        print(f"LabelsManager: set selected class = {iclass}")
        for iB, button in enumerate(self._buttons):
            if iB == self._selected_class:  button.layout = {'border': '3px solid #FFFF00'}
            else:                           button.layout = {'border': '1px solid darkkhaki'}

    def gui( self ) -> ipw.DOMWidget:
        if self.wSelectedClass is None:
            for iC, (color, label) in enumerate(zip( self._colors, self._labels )):
                button = ipw.Button( description=label, layout=ipw.Layout( flex='1 1 auto', height="auto"), border= '1px solid dimgrey'  )
                button.style.button_color = color
                button.on_click( partial( self.set_selected_class, iC ) )
                self._buttons.append( button )
            self.wSelectedClass = ipw.HBox( self._buttons )
            self.wSelectedClass.layout = ipw.Layout( flex='1 1 auto', width = "100%"  )
            self.set_selected_class( 0 )
        return self.wSelectedClass

    def flow(self) -> Optional[ActivationFlow]:
        return self._flow

    def addMarkerAction(self, source: str, marker: Marker, **kwargs ):
        top_marker = self.topMarker
        if not marker.isEmpty() and ( marker != top_marker ):
            self.clearTransientMarkers(marker)
            lgm().log( f"LabelsManager.addMarker: {marker}")
            self.addAction("mark", source, marker=marker )

    def addAction(self, type: str, source: str, **kwargs ):
        from spectraclass.gui.control import UserFeedbackManager, ufm
        ufm().clear()
        new_action = Action(type, source, **kwargs)
        lgm().log(f"ADD ACTION: {new_action}")
        repeat_color = (type == "color") and self.hasActions and (self.topAction.type == "color")
        if not repeat_color:
            self._actions.append( new_action )

    @property
    def hasActions(self) -> bool:
        return len(self._actions) > 0

    @property
    def topAction(self) -> Optional[Action]:
        try:        return  self._actions[-1]
        except:     return None

    @property
    def topMarker(self) -> Optional[Marker]:
        try:        return self.markers[-1]
        except:     return None

    @exception_handled
    def popAction(self) -> Optional[Action]:
        if len(self._actions) > 0:
            action =  self._actions.pop()
            print( f"POP ACTION: {action}, #Actions remainign = {len(self._actions)}" )
            return action

    @property
    def classification(self) -> np.ndarray:
        return self._flow.C

    def initLabelsData( self, point_data: xa.DataArray = None ):
        if point_data is not None:
            self.template = point_data[:,0].squeeze( drop=True )
            self.template.attrs = point_data.attrs
        if self.template is not None:
            self._labels_data: xa.DataArray = xa.full_like( self.template, 0, dtype=np.dtype(np.int32) ).where( self.template.notnull(), self._nodata_value )
            self._labels_data.attrs['_FillValue'] = self._nodata_value
            self._labels_data.name = self.template.attrs['dsid'] + "_labels"
            self._labels_data.attrs[ 'long_name' ] = [ "labels" ]

    @property
    def markers(self):
        return [ a["marker"] for a in self._actions if a.type == "mark" ]

    def getMarker( self, pid: int ) -> Optional[Marker]:
        for marker in self.markers:
            if pid in marker.pids: return marker
        return None

    def updateLabels(self):
        self._labels_data = xa.where( self.template.notnull(), 0, self._nodata_value )
        for marker in self.markers:
            for pid in marker.pids:
                self._labels_data[ pid ] = marker.cid
        self.log_markers("updateLabels")
        lgm().log(f" #LABELS: { np.count_nonzero( self._labels_data.data > 0 ) }" )

    def labels_data( self ) -> xa.DataArray:
        self.updateLabels()
        return self._labels_data.copy( self._optype == "distance" )

    def log_markers(self, msg: str ):
        log_strs = []
        for m in self.markers:
            log_strs.append( f"[{m.cid}:{m.pids[0]}]" if m.size == 1 else  f"M{m.cid}-{m.size}" )
        lgm().log( f"  ----------------------------> log_markers[{msg}]: {' '.join(log_strs)}")

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

    def clearMarkers(self):
        self._actions = []
        self.initLabelsData()

    def refresh(self):
        self.clearMarkers()

    def deletePid(self, pid: int ):
        for marker in self.markers: marker.deletePid( pid )

    # def clearMarkerConflicts(self, m: Marker):
    #     markers = []
    #     for marker in self._markers:
    #         if marker.cid != m.cid:
    #             marker.deletePids( marker.pids )
    #         if (not marker.isEmpty()):
    #             markers.append( marker )
    #     self._markers = markers

    def clearTransientMarkers(self, m: Marker):
        if len(self.markers) > 0:
            top_marker = self.topMarker
            if (top_marker.cid == 0) and (not m.isEmpty()):
                self.clearMarker( top_marker )

    def clearMarker( self, m ):
        for action in reversed(self._actions):
            if m == action["marker"]:
                self._actions.remove(action)
                return

    @property
    def currentMarker(self) -> Marker:
        marker = self.markers[ -1 ] if len( self.markers ) else None
        return marker

    def getPids( self, cid = -1 ) -> List[int]:
        pids = []
        icid =  self.current_cid if cid < 0 else cid
        for m in self.markers:
            if (icid == m.cid): pids.extend( m.pids )
        return pids

    def getMarkedPids( self ) -> List[int]:
        pids = []
        for m in self.markers:
            if (m.cid > 0): pids.extend( m.pids )
        return pids

    def getLabelMap( self ) -> Dict[int,Set[int]]:
        label_map = {}
        for m in self.markers:
            pids = label_map.get( m.cid, set() )
            label_map[m.cid] = pids.union( set(m.pids) )
        return label_map

    @property
    def selectedLabel(self):
        return self._labels[ self.current_cid ]

    def selectedColor(self, mark: bool ) -> Tuple[int,List[float]]:
        icolor = self.current_cid if mark else 0
        return icolor, self._colors[ icolor ]

    @property
    def colors(self)-> List[Tuple]:
        return self._colors

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def nLabels(self) -> int:
        return len(self._labels)

    def setLabels(self, labels: List[Tuple[str, str]], **kwargs):
        unlabeled_color = kwargs.get( 'unlabeled', "YELLOW" )
        label_list = [ ('Unlabeled', unlabeled_color ) ] + labels
        for ( label, color ) in labels:
            if color.upper() == unlabeled_color: raise Exception( f"{unlabeled_color} is a reserved color")
        self._colors = [ item[1] for item in label_list ]
        self._labels = [ item[0] for item in label_list ]

    def getSeedPointMask(self) -> xa.DataArray:
        from spectraclass.gui.control import UserFeedbackManager, ufm
        if self.currentMarker is None:
            ufm().show( "Error: Must Label some points before executing this operation!", "red" )
            return xa.DataArray( np.empty(shape=[0], dtype=np.int) )
        else:
            from spectraclass.data.base import DataManager
            model_data: xa.DataArray = DataManager.instance().getModelData()
            seed_points = xa.full_like( model_data[:, 0], 0, np.dtype(np.int32) )
            seed_points[ self.currentMarker.pids ] = 1
            return seed_points

    @exception_handled
    def mark_points( self, point_ids: np.ndarray = None, cid: int = -1 ):
        from spectraclass.gui.control import UserFeedbackManager, ufm
        icid: int = cid if cid > -1 else self.current_cid
        if point_ids is None:
            if self.currentMarker is None:
                ufm().show("Must select point(s) to mark.", "red")
                return
            self.currentMarker.cid = icid
            point_ids = self.currentMarker.pids
        else:
            lgm().log( f" LM: mark_points -> npts = {point_ids.size}, id range = {[point_ids.min(), point_ids.max()]}")

        new_pids: np.ndarray = self.getNewPids( point_ids, icid )
        self.addMarkerAction(  "points", Marker( new_pids, cid ) )
        return self.current_cid

    def getNewPids(self, point_ids: np.ndarray, cid: int ) -> np.ndarray:
        current_pids: np.ndarray = np.array( self.getPids( cid ) )
        if len(current_pids) == 0:
            return point_ids
        elif point_ids.size == 1:
            new_pids = [] if point_ids[0] in current_pids else point_ids
            return np.array( new_pids )
        else:
            shared_values_mask = np.isin( point_ids, current_pids, assume_unique=True )
            return point_ids[ np.invert( shared_values_mask ) ]


