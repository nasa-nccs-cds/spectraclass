from collections import OrderedDict
from typing import List, Union, Dict, Callable, Tuple, Optional, Any, Set
from spectraclass.data.spatial.tile.tile import Block
import os, collections.abc
from functools import partial
import panel as pn
from ..graph.manager import ActivationFlow
import traitlets.config as tlc
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
from spectraclass.model.base import SCSingletonConfigurable
from spectraclass.gui.spatial.widgets.markers import Marker
import xarray as xa
import numpy as np

RIGHT_BUTTON = 3
MIDDLE_BUTTON = 2
LEFT_BUTTON = 1

def c2rgb( color: Union[str,List] ) -> Tuple[float,float,float]:
    from matplotlib import colors
    if isinstance(color, str):  return colors.to_rgb(color)
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
        self._colors: List[str] = []
        self._labels: List[str] = []
        self._indices: List[int] = []
        self._markers: List[Marker] = []
        self._labels_data: xa.DataArray = None
        self._flow: ActivationFlow = None
        self._actions: List[Action] = []
        self._label_maps: List[np.ndarray] = []
        self.class_selector: pn.widgets.RadioButtonGroup = None
        self._nodata_value = -1
        self._optype = None
        self.template = None
        self.n_spread_iters = 1
#        self.wSelectedClass: ipw.HBox = None
#        self.get_rgb_colors = np.vectorize(self.get_rgb_color)
        self._buttons = []
        self.unlabeled_color =  "white"
        self.unlabeled_index =  9999

    def get_label( self, cid: int ) -> str:
        idx = self._indices.index( cid )
        return self.labels[idx]

    def set_classification( self, classification: np.ndarray ):
        crange = [ classification.min(), classification.max() ]
        lgm().log( f"set_classification: shape={classification.shape}, range={crange}" )
        self._classification = classification

    def clear_classification( self ):
        self._classification = None

    @property
    def current_class(self) -> str:
        return self.class_selector.value

    @property
    def _idx(self) -> int:
        return self._labels.index( self.current_class )

    @property
    def current_cid(self) -> int:
        return self._indices[ self._idx ]

    @property
    def current_color(self) -> str:
        return self._colors[  self._idx  ]

    def get_rgb_color( self, cid: int, probe: bool = False ) -> Tuple[float,float,float]:
        from matplotlib import colors
        idx = self._indices.index( cid )
        color = self.unlabeled_color if probe else self._colors[ idx ]
        return colors.to_rgb( color )

    def get_rgb_colors(self, cids: List[int], probe: bool = False ) -> np.ndarray:
        cdata = np.array( [ self.get_rgb_color(cid,probe) for cid in cids ] ) * 255.0
        return cdata.astype(np.uint8)

    # def set_selected_class(self, iclass, *args ):
    #     from spectraclass.gui.control import UserFeedbackManager, ufm
    #     from spectraclass.application.controller import app
    #     ufm().clear()
    #     self.current_cid = iclass
    #     for iB, button in enumerate(self._buttons):
    #         if iB == self.current_cid:  button.layout = {'border': '3px solid #FFFF00'}
    #         else:                           button.layout = {'border': '1px solid darkkhaki'}
    #     app().update_current_class( iclass )

    # def gui( self ) -> ipw.DOMWidget:
    #     pn.widgets.RadioButtonGroup(name='Class Selection', value=lm().labels[0], options=lm().labels)
    #
    #     if self.wSelectedClass is None:
    #         for iC, (color, label) in enumerate(zip( self._colors, self._labels )):
    #             button = ipw.Button( description=label, layout=ipw.Layout( width = "100%", max_width="500px" ), border= '1px solid dimgrey'  ) # flex='1 1 auto',
    #             button.style.button_color = color
    #             button.on_click( partial( self.set_selected_class, iC ) )
    #             self._buttons.append( button )
    #         self.wSelectedClass = ipw.HBox( self._buttons, layout = ipw.Layout( width = "100%"  ) )
    #         self.set_selected_class( 0 )
    #     return self.wSelectedClass


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
        from spectraclass.data.spatial.tile.manager import tm
        self.clearTransientMarkers(marker)
        for m in self._markers:
            m.clear_gids(marker.gids)
        self._markers.append( marker )
        lgm().log(f"LabelsManager[{tm().image_index}:{tm().block_index}].addMarker: cid={marker.cid}, #pids={len(marker.gids)}, active = {marker.active()}, block={marker.block_index}, image={marker.image_index}")

    def popMarker(self, mtype: str = None ) -> Optional[Marker]:
        for iM in range( len(self._markers)-1, -1, -1 ):
            if (mtype is None) or (self._markers[iM].type == mtype):
                return self._markers.pop(iM)

    @property
    def markers(self):
        return [m for m in self._markers if m.active()]

    def getMarkers(self) -> List[Marker]:
        return self._markers

    def getPoints(self) -> List[Tuple[float,float,str]]:
        points = []
        for m in self._markers:
            point = m.props['point']
            cname = self._labels[ m.cid ]
            points.append( (point[0], point[1], cname) )
        return points

    def addAction(self, type: str, source: str, **kwargs ):
        new_action = Action(type, source, **kwargs)
        lgm().log(f"ADD ACTION: {new_action}")
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
            self.template = point_data[:,0] # .squeeze( drop=True ) # if point_data.ndim == 2 else point_data
            self.template.attrs = point_data.attrs
        if self.template is not None:
            self._labels_data: xa.DataArray = xa.full_like( self.template, 0, dtype=np.int32 ).where( self.template.notnull(), nodata_value )
            self._labels_data.attrs['_FillValue'] = nodata_value
            self._labels_data.name = self.template.attrs['dsid'] + "_labels"
            self._labels_data.attrs[ 'long_name' ] =  "labels"

    def _init_data(self):
        from ..graph.manager import ActivationFlowManager
        from spectraclass.data.spatial.tile.manager import tm
        if self._flow is None:
            point_data: xa.DataArray = tm().getBlock().getPointData()[0]
            self._init_labels_data( point_data )
            self._flow = ActivationFlowManager.instance().getActivationFlow()

    def getMarker( self, pid: int ) -> Optional[Marker]:
        lgm().log( f" ^^^^^^^^^ getMarker[{pid}] -> markers = {self.markers}")
        for marker in self.markers:
            if pid in marker.gids: return marker
        return None

    def log_markers(self, msg: str ):
        log_strs = []
        for m in self.markers:
            log_strs.append( f"[{m.cid}:{m.gids[0]}]" if m.size == 1 else f"M{m.cid}-{m.size}")
        lgm().log( f"  ----------------------------> log_markers[{msg}]: {' '.join(log_strs)}")

    def updateLabels(self):
        self._init_data()
        mks: List[Marker] = self.markers
        lgm().log( f" NMarkers = {len(mks)}")
        self._labels_data[:] = 0
        for marker in mks:
            lgm().log(f" MARKER[{marker.cid}]: #pids = {len(marker.gids)}")
            self._labels_data.loc[ dict(samples=marker.gids)] = marker.cid

    def getTrainingLabels(self) -> Dict[ Tuple, np.ndarray ]:
        label_data = {}
        for marker in self._markers:
            key = ( marker.image_index, marker.block_coords, marker.cid )
            label_data[key] = marker.gids if (key not in label_data) else np.append(label_data[key], marker.gids, axis=0)
        return label_data

    def getTrainingBlocks(self) -> List[Block]:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block_data = { ( marker.image_index, marker.block_coords ) for marker in self._markers }
        lgm().log( f" ----------------------------->>>  getTrainingBlocks: block_data= {block_data}, #markers={len(self._markers)}" )
        return [ tm().getBlock(tindex=tindex, block_coords=block_coords) for (tindex,block_coords) in block_data ]

    def getLabelsArray(self) -> xa.DataArray:
        self.updateLabels()
        return self._labels_data.copy()

    def getLabelDataset(self) -> xa.Dataset:
        data_arrays = {}
        labeled_blocks: List[Block] = self.getTrainingBlocks()
        for block in labeled_blocks:
            lname = f"labels-{block.tile_index}-{block.block_coords[0]}-{block.block_coords[1]}"
            data_arrays[ lname ] = self.get_label_map( block=block )
        label_dset = xa.Dataset( data_arrays )
        return label_dset

    @exception_handled
    def loadLabelData( self, labels_dset: Union[str,bool] ):
        from spectraclass.data.base import DataManager, dm
        if isinstance(labels_dset, str): dm().labels_dset = labels_dset
        lgm().log( f'Loading labels file: {dm().labels_file}', print=True )
        labels_dset: xa.Dataset = xa.open_dataset( dm().labels_file )
        for (vid, labels_var) in labels_dset.data_vars.items():
            ( _, image_idx, bidx0, bidx1 ) = vid.split('-')
            point_index = np.arange(0, labels_var.shape[-1] * labels_var.shape[-2])
            lgm().log(f'Loading labels for image-{image_idx}: block={(bidx0,bidx1)} ')
            for cid in range( 1, lm().nLabels ):
                label_mask = (labels_var == cid).values.flatten()
                if np.count_nonzero( label_mask ) > 0:
                    pids = point_index[ label_mask ]
                    bindex = (int(bidx0),int(bidx1))
                    marker = Marker( 'marker', pids, cid, block_index=bindex, image_index=int(image_idx) )
                    self.addMarker( marker )

    def saveLabelData( self, lid: str = None, **kwargs ) -> xa.Dataset:
        from spectraclass.gui.control import UserFeedbackManager, ufm
        from spectraclass.data.base import DataManager, dm
        if lid is not None: dm().labels_dset = lid
        label_dset = self.getLabelDataset()
        label_dset.to_netcdf( dm().labels_file )
        ufm().show( f"Saving labels to file: {dm().labels_file}")
        return label_dset

    def getClassification(self) -> Optional[xa.DataArray]:
        from spectraclass.learn.manager import ClassificationManager, cm
        from spectraclass.data.spatial.tile.manager import tm
        if cm().classification is None:
            return None
        elif cm().classification.dims[0] == 'samples':
            return cm().classification
        else:
            class_data, _, _ = tm().getBlock().raster2points( cm().classification )
            return class_data

    @exception_handled
    def graphLabelData(self):
        from spectraclass.gui.lineplots.manager import GraphPlotManager, gpm, LinePlot
        from spectraclass.data.spatial.tile.manager import tm
        block = tm().getBlock()
        graph: LinePlot = gpm().current_graph()
        class_data: Optional[xa.DataArray] = self.getClassification()
        if class_data is not None:
            for cid in self.get_cids():
                classmask: np.ndarray = (class_data.values.flatten() == cid)
                lgm().log(f"graphLabelData: cid={cid}, #pids={np.count_nonzero(classmask)}, dims={list(class_data.dims)}, coords={list(class_data.coords.keys())}")
                samples: xa.DataArray = class_data.coords['samples']
                pids: np.ndarray = samples.values[classmask]
                graph.addMarker( Marker("labels", pids, cid) )

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
        lgm().log( "CLEAR ALL MARKERS")
        self.clearMarkers()

    def deletePids(self, pids: List[int] ):
        for pid in pids: self.deletePid( pid )

    def deletePid(self, pid: int ):
        if pid >= 0 :
            empty_markers = []
            for marker in self.markers:
                marker.clear_gid(pid)
                if marker.empty: empty_markers.append( marker )
            for m in empty_markers:
                lgm().log( f"LM: Removing marker: {m}")
                self._markers.remove( m )

    def clearTransientMarkers(self, m: Marker):
        top_marker = self.topMarker
        if top_marker and (top_marker.cid == 0) and (not m.empty):
            self.clearMarker( top_marker )

    def clearMarker( self, m ):
        for marker in reversed(self._markers):
            lgm().log(f"clearMarker test: {marker} <-> {m}")
            if m == marker:
                lgm().log(f"LM: Removing marker: {m}")
                self._markers.remove( marker )
                return

    @property
    def currentMarker(self) -> Marker:
        marker = self.markers[ -1 ] if len( self.markers ) else None
        return marker

    def getGids( self, cid = -1 ) -> List[int]:
        gids = []
        icid =  self.current_cid if cid < 0 else cid
        for m in self.markers:
            if (icid == m.cid): gids.extend(m.gids)
        return gids

    def getMarkedGids( self ) -> List[int]:
        gids = []
        for m in self.markers:
            if (m.cid > 0): gids.extend(m.gids)
        return gids

    def getLabelMap( self, update_directory_table = False ) -> Dict[int,Set[int]]:
        from spectraclass.gui.unstructured.table import tbm
        label_map = {}
        if update_directory_table: tbm().clear_table(0)
        for m in self.markers:
            pids = label_map.get( m.cid, set() )
            label_map[m.cid] = pids.union(set(m.gids))
            for cid, lmap in label_map.items():
                if (cid > 0) and (cid != m.cid):
                    common_items = lmap.intersection(m.gids)
                    if len( common_items ):
                        label_map[cid] = lmap.difference(common_items)
            if update_directory_table:
                tbm().edit_table(0, m.gids, "cid", m.cid)
        return label_map

    def get_label_data( self ) -> Dict[int,Set[int]]:
        label_map = {}
        for m in self.markers:
            pids = label_map.get( m.cid, set() )
            label_map[m.cid] = pids.union(set(m.gids))
        return label_map

    def get_cids( self ) -> Set[int]:
        return set( [m.cid for m in self.markers] )

    def get_label_map( self, **kwargs ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block = kwargs.get( 'block', tm().getBlock() )
        mtype = kwargs.get( 'type' )
        projected = kwargs.get( 'projected', True )
        xcmap: xa.DataArray = block.classmap()
        cmap = xcmap.values.copy()
        markers = self.getMarkers()
        lgm().log( f" *GET LABEL MAP: {len(markers)} markers")
        for marker in markers:
            if marker.relevant(mtype,block=block):
                if marker.mask is not None:
                    fmask = marker.mask.flatten()
                    lgm().log(f" Setting {np.count_nonzero(fmask)} labels for cid = {marker.cid}, block={block.index}, WITH  MASK")
                    np.ravel(cmap)[ fmask ] = marker.cid
                else:
                    lgm().log( f" Setting {len(marker.gids)} labels for cid = {marker.cid}, block={block.index}, NO MASK, projected= {projected}")
                    for gid in marker.gids:
                        if projected:
                            idx = block.gid2indices( gid )
                            cmap[ idx['iy'], idx['ix'] ] = marker.cid
                        else:
                            pass
        lgm().log(f" get_label_map, #labeled points = {np.count_nonzero(cmap)}")
        return xcmap.copy(data=cmap)

 #   from spectraclass.data.spatial.tile.manager import tm
 #   block = tm().getBlock()
 #   block.getSelectedPoint(self, cy: float, cx: float )

    @log_timing
    def update_label_map( self, mask: xa.DataArray, cid: int,  **kwargs ) -> xa.DataArray:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        block = kwargs.get( 'block', tm().getBlock() )
        cmap: xa.DataArray = block.classmap()
        lgm().log( f" **UPDATE LABEL MAP: {len(self.markers)} markers")
        for marker in self.markers:
            if marker.type not in ["cluster"]:
                lgm().log( f"update_label_map->MARKER[{marker.type}]: Setting {len(marker.gids)} labels for cid = {marker.cid}")
         # -->       points2raster
                for pid in marker.gids:
                    idx = block.gid2indices(pid)
                    cmap[ idx['iy'], idx['ix'] ] = marker.cid
        cmap[ mask ] = cid
        return cmap

    @property
    def selectedLabel(self):
        return self._labels[ self._idx ]

    def selectedColor(self, mark: bool ) -> Tuple[int,str]:
        icolor =  self._idx if mark else self.unlabeled_index
        return self._indices[icolor], self._colors[ icolor ]

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

    def setLabels(self, labels: Union[List[Tuple[str, str]],Dict[int,Tuple[str, str]]], **kwargs):
        load_existing = kwargs.get('load', False)
        self.unlabeled_color = kwargs.get('unlabeled_color', "white")
        selected = kwargs.get('selected', 0)

        if type(labels) == list:
            self.unlabeled_index = kwargs.get( 'unlabeled_index', 0 )
            label_list = [ ('Unlabeled', self.unlabeled_color ) ] + labels
            for ( label, color ) in labels:
                if color.lower() == self.unlabeled_color: raise Exception( f"{self.unlabeled_color} is a reserved color")
            self._colors = [ item[1] for item in label_list ]
            self._labels = [ item[0] for item in label_list ]
            self._indices = list(range(len(label_list)))
        elif type(labels) == dict:
            self.unlabeled_index = kwargs.get('unlabeled_index', 9999)
            for ( label, color ) in labels.values():
                if color.lower() == self.unlabeled_color: raise Exception( f"{self.unlabeled_color} is a reserved color")
            label_selections = { self.unlabeled_index: ('Unlabeled', self.unlabeled_color ) }
            label_selections.update( labels )
            for index, (label,color) in label_selections.items():
                self._colors.append( color )
                self._labels.append( label )
                self._indices.append( index )

        self.class_selector = pn.widgets.RadioButtonGroup(name='Class Selection', value=self._labels[0], options=self._labels)
        if load_existing:
            self.loadLabelData( load_existing )

    @property
    def labelmap(self) -> Dict[str,str]:
        return { k:v for (k,v) in zip(self._labels,self._colors) }

    def getSeedPointMask(self) -> xa.DataArray:
        from spectraclass.gui.control import UserFeedbackManager, ufm
        if self.currentMarker is None:
            ufm().show( "Error: Must Label some points before executing this operation!", "warning" )
            return xa.DataArray( np.empty(shape=[0], dtype=np.int32) )
        else:
            from spectraclass.data.base import DataManager
            model_data: xa.DataArray = DataManager.instance().getModelData()
            seed_points = xa.full_like( model_data[:, 0], 0, np.dtype(np.int32) )
            seed_points[ self.currentMarker.gids] = 1
            return seed_points

    @exception_handled
    def mark_points( self, gids: np.ndarray, cid: int, type: str = "markers" ) -> Optional[Marker]:
        from spectraclass.gui.control import UserFeedbackManager, ufm
        from spectraclass.application.controller import app
        from spectraclass.gui.spatial.widgets.markers import Marker
        icid: int = cid if cid > -1 else self.current_cid
        if gids is None:
            if self.currentMarker is None:
                lgm().log( f" LM: mark_points -> NO POINTS SELECTED")
                ufm().show("Must select point(s) to mark.", "red")
                return None
            self.currentMarker.cid = icid
            gids = self.currentMarker.gids

        lgm().log( f" LM: mark_points -> npts = {gids.size}, id range = {[gids.min(), gids.max()]}")
        marker = Marker( type, gids, icid )
        app().add_marker( marker )
        return marker

    def getNewGids(self, gids: np.ndarray, cid: int) -> np.ndarray:
        current_gids: np.ndarray = np.array( self.getGids( cid ) )
        if len(current_gids) == 0:
            return gids
        elif gids.size == 1:
            new_pids = [] if gids[0] in current_gids else gids
            return np.array( new_pids )
        else:
            shared_values_mask = np.isin(gids, current_gids, assume_unique=True)
            return gids[ np.invert(shared_values_mask)]

    @property
    def block(self) -> Block:
        from spectraclass.data.spatial.tile.manager import TileManager, tm
        return tm().getBlock()

    def on_button_press(self, x, y, button = LEFT_BUTTON ):
        from spectraclass.gui.control import ufm
        if (x != None) and (y != None) :
            gid, ix, iy = self.block.coords2gid(y, x)
            marker = Marker( "marker", [gid], self.current_cid, point=(x,y) )
            if int(button) == RIGHT_BUTTON:
                self.clearMarker( marker )
            elif int(button) == LEFT_BUTTON:
                lgm().log(f" >> selected gid = {gid}, button = {button}")
                ufm().show( f" event[{x:.2f},{y:.2f}]: ({ix},{iy},{gid})" )
                self.addMarker( marker )


