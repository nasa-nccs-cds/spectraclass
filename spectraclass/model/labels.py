from collections import OrderedDict
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Set
from spectraclass.data.spatial.tile.tile import Block
import collections.abc
from functools import partial
import ipywidgets as ipw
import matplotlib.colors as mcolors
from ..graph.manager import ActivationFlow
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.gui.spatial.widgets.markers import Marker
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

def get_color_bounds( color_values: List[float] ) -> List[float]:
    color_bounds = []
    for iC, cval in enumerate( color_values ):
        if iC == 0: color_bounds.append( cval - 0.5 )
        else: color_bounds.append( (cval + color_values[iC-1])/2.0 )
    color_bounds.append( color_values[-1] + 0.5 )
    return color_bounds

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
        self._colors: List[str] = None
        self._labels = None
        self._markers: List[Marker] = []
        self._labels_data: xa.DataArray = None
        self._flow: ActivationFlow = None
        self._actions: List[Action] = []
        self._label_maps: List[np.ndarray] = []
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

    @property
    def current_color(self) -> str:
        return self._colors[ self._selected_class ]

    def set_selected_class(self, iclass, *args ):
        from spectraclass.gui.control import UserFeedbackManager, ufm
        from spectraclass.application.controller import app
        ufm().clear()
        self._selected_class = iclass
        for iB, button in enumerate(self._buttons):
            if iB == self._selected_class:  button.layout = {'border': '3px solid #FFFF00'}
            else:                           button.layout = {'border': '1px solid darkkhaki'}
        app().update_current_class( iclass )

    def gui( self ) -> ipw.DOMWidget:
        if self.wSelectedClass is None:
            for iC, (color, label) in enumerate(zip( self._colors, self._labels )):
                button = ipw.Button( description=label, layout=ipw.Layout( width = "100%", max_width="500px" ), border= '1px solid dimgrey'  ) # flex='1 1 auto',
                button.style.button_color = color
                button.on_click( partial( self.set_selected_class, iC ) )
                self._buttons.append( button )
            self.wSelectedClass = ipw.HBox( self._buttons, layout = ipw.Layout( width = "100%"  ) )
            self.set_selected_class( 0 )
        return self.wSelectedClass

    def get_labels_colormap(self):
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        import matplotlib as mpl
        rgbs = [cval[2] for cval in self.labeledColors]
        cmap: ListedColormap = ListedColormap(rgbs)
        color_values = [float(cval[0]) for cval in self.labeledColors]
#        color_bounds = get_color_bounds(color_values)
        ncolors = len(self.labeledColors)
        color_bounds = np.linspace( -0.5, ncolors-0.5, ncolors+1 )
        norm = mpl.colors.BoundaryNorm( color_bounds, ncolors, clip=True )
        lgm().log( f"labels_colormap: colors={rgbs}, color_values={color_values}, color_bounds={color_bounds}, ncolors={ncolors}")
        result =  dict( cmap=cmap, norm=norm, boundaries=color_bounds, ticks=color_values, spacing='proportional')
        return result

    def flow(self) -> Optional[ActivationFlow]:
        return self._flow

    def setLabelData( self, labels_map: np.ndarray ):
        self._label_maps.append( labels_map.copy() )

    def undoLabelsUpdate(self) -> np.ndarray:
        if len( self._label_maps ) > 1:
            self._label_maps.pop()
        return self._label_maps[-1]

    def clearLabels(self) -> np.ndarray:
        if len( self._label_maps ) > 1:
            self._label_maps = self._label_maps[:1]
        return self._label_maps[-1]

    def addMarker(self, marker: Marker ):
        self.clearTransientMarkers(marker)
        self._markers.append( marker )

    @property
    def markers(self):
        return self._markers

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
        try:        return self._markers[-1]
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

    def _init_labels_data(self, point_data: xa.DataArray = None):
        nodata_value = -1
        if point_data is not None:
            self.template = point_data[:,0].squeeze( drop=True )
            self.template.attrs = point_data.attrs
        if self.template is not None:
            lgm().log( f" *** Init Labels Data, dims = {self.template.dims}, shape = {self.template.shape}")
            self._labels_data: xa.DataArray = xa.full_like( self.template, 0, dtype=np.int32 ).where( self.template.notnull(), nodata_value )
            self._labels_data.attrs['_FillValue'] = nodata_value
            self._labels_data.name = self.template.attrs['dsid'] + "_labels"
            self._labels_data.attrs[ 'long_name' ] =  "labels"

    def _init_data(self):
        from ..graph.manager import ActivationFlowManager
        from spectraclass.data.base import DataManager
        if self._flow is None:
            project_data: xa.Dataset = DataManager.instance().loadCurrentProject("labels")
            point_data: xa.DataArray = project_data["plot-y"]
            self._init_labels_data( point_data )
            self._flow = ActivationFlowManager.instance().getActivationFlow()

    def getMarker( self, pid: int ) -> Optional[Marker]:
        lgm().log( f" ^^^^^^^^^ getMarker[{pid}] -> markers = {self.markers}")
        for marker in self.markers:
            if pid in marker.pids: return marker
        return None

    def log_markers(self, msg: str ):
        log_strs = []
        for m in self.markers:
            log_strs.append( f"[{m.cid}:{m.pids[0]}]" if m.size == 1 else  f"M{m.cid}-{m.size}" )
        lgm().log( f"  ----------------------------> log_markers[{msg}]: {' '.join(log_strs)}")

    def updateLabels(self):
        self._init_data()
        mks: List[Marker] = self.markers
        lgm().log( f" NMarkers = {len(mks)}")
        self._labels_data[:] = 0
        for marker in mks:
            lgm().log(f" MARKER[{marker.cid}]: #pids = {len(marker.pids)}")
            self._labels_data[ marker.pids ] = marker.cid

    def getLabelsArray(self) -> xa.DataArray:
        self.updateLabels()
        return self._labels_data.copy()

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
        self._markers = []
        self._init_labels_data()

    def refresh(self):
        self.clearMarkers()

    def deletePid(self, pid: int ):
        empty_markers = []
        for marker in self._markers:
            marker.deletePid(pid)
            if marker.empty: empty_markers.append( marker )
        for m in empty_markers:
            lgm().log( f"LM: Removing marker: {m}")
            self._markers.remove( m )


    # def clearMarkerConflicts(self, m: Marker):
    #     points_selection = []
    #     for marker in self._markers:
    #         if marker.cid != m.cid:
    #             marker.deletePids( marker.pids )
    #         if (not marker.isEmpty()):
    #             points_selection.append( marker )
    #     self._markers = points_selection

    def clearTransientMarkers(self, m: Marker):
        top_marker = self.topMarker
        if top_marker and (top_marker.cid == 0) and (not m.empty):
            self.clearMarker( top_marker )

    def clearMarker( self, m ):
        for marker in reversed(self._markers):
            if m == marker:
                self._markers.remove( marker )
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

    def getLabelMap( self, update_directory_table = False ) -> Dict[int,Set[int]]:
        from spectraclass.gui.unstructured.table import tm
        label_map = {}
        if update_directory_table: tm().clear_table(0)
        for m in self.markers:
            pids = label_map.get( m.cid, set() )
            label_map[m.cid] = pids.union( set(m.pids) )
            for cid, lmap in label_map.items():
                if (cid > 0) and (cid != m.cid):
                    common_items = lmap.intersection(m.pids)
                    if len( common_items ):
                        label_map[cid] = lmap.difference(common_items)
            if update_directory_table:
                tm().edit_table( 0, m.pids, "cid", m.cid )
        return label_map

    def get_label_data( self ) -> Dict[int,Set[int]]:
        label_map = {}
        for m in self.markers:
            pids = label_map.get( m.cid, set() )
            label_map[m.cid] = pids.union( set(m.pids) )
        return label_map

    @log_timing
    def get_label_map(self, block: Block, type: str = None ) -> xa.DataArray:
        cmap: xa.DataArray = block.classmap()
        for marker in lm().markers:
            if (type is None) or (marker.type == type):
                lgm().log( f"\nSetting {marker.pids.size} labels for cid = {marker.cid}")
                for pid in marker.pids:
                    idx = block.pindex2indices( pid )
                    cmap[ idx['iy'], idx['ix'] ] = marker.cid
        return cmap

    @property
    def selectedLabel(self):
        return self._labels[ self.current_cid ]

    def selectedColor(self, mark: bool ) -> Tuple[int,str]:
        icolor = self.current_cid if mark else 0
        return icolor, self._colors[ icolor ]

    @property
    def colors(self)-> List[str]:
        return self._colors

    @property
    def graph_colors(self)-> List[str]:
        return [ 'black' ] + self._colors[1:]

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def nLabels(self) -> int:
        return len(self._labels)

    @property
    def labeledColors(self) -> List[Tuple[int,str,str]]:
        values = range(len(self._colors))
        return list(zip(values, self._labels, self._colors))

    def setLabels(self, labels: List[Tuple[str, str]], **kwargs):
        unlabeled_color = kwargs.get( 'unlabeled', "white" )
        label_list = [ ('Unlabeled', unlabeled_color ) ] + labels
        for ( label, color ) in labels:
            if color.lower() == unlabeled_color: raise Exception( f"{unlabeled_color} is a reserved color")
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
    def mark_points( self, point_ids: np.ndarray, cid: int, type: str = "markers" ):
        from spectraclass.gui.control import UserFeedbackManager, ufm
        from spectraclass.gui.spatial.widgets.markers import Marker
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
        self.addMarker( Marker( type, new_pids, cid ) )
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


