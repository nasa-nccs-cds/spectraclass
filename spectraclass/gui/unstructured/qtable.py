import qgrid
from typing import List, Dict, Set
from spectraclass.util.logs import lgm
from functools import partial
import xarray as xa
import numpy as np
import pandas as pd
import ipywidgets as ipw
from spectraclass.widgets.buttons import ToggleButton
x
import traitlets as tl
from spectraclass.model.labels import LabelsManager
from spectraclass.model.base import SCSingletonConfigurable

class TableManager(SCSingletonConfigurable):

    def __init__(self):
        super(TableManager, self).__init__()
        self._wGui: ipw.VBox = None
        self._dataFrame: pd.DataFrame = None
        self._cols: List[str] = None
        self._tables: List[qgrid.QgridWidget] = []
        self._wTablesWidget: ipw.Tab = None
        self._current_column_index: int = 0
        self._current_selection: List[int] = []
        self._class_map = None
        self._search_widgets = None
        self._match_options = {}
        self._events = []
        self._broadcast_selection_events = True
        self.mark_on_selection = False

    def init(self, **kwargs):
        catalog: Dict[str,np.ndarray] = kwargs.get( 'catalog', None )
        project_data: xa.Dataset = dm().loadCurrentProject("table")
        if catalog is None:  catalog = { tcol: project_data[tcol].values for tcol in dm().modal.METAVARS }
        nrows = catalog[ dm().modal.METAVARS[0] ].shape[0]
        lgm().log( f"Catalog: nrows = {nrows}, entries: {[ f'{k}:{v.shape}' for (k,v) in catalog.items() ]}" )
        self._dataFrame: pd.DataFrame = pd.DataFrame( catalog, dtype='U', index=pd.Int64Index( range(nrows), name="Index" ) )
        self._cols = list(catalog.keys()) + [ "cid" ]
        self._class_map = np.zeros( nrows, np.int32 )
        self._flow_class_map = np.zeros( nrows, np.int32 )

    def edit_table(self, cid, index, column, value ):
         table = self._tables[cid]
         table._df.loc[index, column] = value
         table._unfiltered_df.loc[index, column] = value

    def update_table(self, cid, fire_event = True):
        table = self._tables[cid]
        table._update_table( triggered_by='update_table', fire_data_change_event=fire_event )

    def update_selection(self):
        from spectraclass.model.labels import lm
 #       self._broadcast_selection_events = False
        label_map: Dict[int,Set[int]] = lm().getLabelMap()
        directory = self._tables[0]
        changed_pids = dict()
        n_changes = 0
        for (cid,table) in enumerate(self._tables):
            if cid > 0:
                current_pids = set( table.get_changed_df().index.tolist() )
                new_ids: Set[int] = label_map.get( cid, set() )
                deleted_pids = current_pids - new_ids
                added_pids = new_ids - current_pids
                nc = len(added_pids) + len(deleted_pids)
                if nc > 0:
                    n_changes = n_changes + nc
                    lgm().log(f" TM----> update_selection[{cid}]" )
                    if len(deleted_pids): lgm().log(f"    ######## deleted: {deleted_pids} ")
                    if len(added_pids):   lgm().log(f"    ######## added: {added_pids} ")
                    for pid in added_pids:    changed_pids[pid] = cid
                    for pid in deleted_pids:  changed_pids[pid] = 0
                    table.remove_rows( deleted_pids ) # table._remove_rows( deleted_pids )
                    for pid in added_pids:
                        row = directory.get_changed_df().loc[pid].to_dict()
                        row.update( dict( Class=cid, Index=pid ) )
                        lgm().log(f"  **TABLE-> update_selection[{cid},{pid}]: row = {row}")
                        table._add_row( row.items() )
        if n_changes > 0:
            lgm().log(f" TM----> edit directory[ {changed_pids} ]")
            for (pid,cid) in changed_pids.items():
                directory.edit_cell( pid, "cid", cid ) # self.edit_table( 0, polyId, "Class", cid )
#            directory.change_selection([])

#        self._broadcast_selection_events = True

#        self.update_table( 0, False )

        # directory_table = self._tables[0]
        # for (cid, pids) in label_map.items():
        #     table = self._tables[cid]
        #     if cid > 0:
        #         for polyId in pids:
        #             directory_table.edit_cell( polyId, "Class", cid )
        #             self._class_map[polyId] = cid
        #             row = directory_table.df.loc[polyId]
        #             table.add_row( row )
        #
        #         index_list: List[int] = selection_table.index.tolist()
        #         table.edit_cell( index_list, "cid", cid )
        #         lgm().log( f" Edit directory table: set _classes for gindices {index_list} to {cid}")
        #         table.df = pd.concat( [table.df, selection_table] )
        #         lgm().log(f" Edit class table[{cid}]: add pids {pids}, append selection_table with shape {selection_table.shape}")

#     def mark_selection(self):
#         from spectraclass.model.labels import LabelsManager, lm
# #        self._broadcast_selection_events = False
#         label_map: Dict[int,Set[int]] = lm().getLabelMap()
#         directory = self._tables[0]
#         changed_pids = dict()
#         n_changes = 0
#         for (cid,table) in enumerate(self._tables):
#             if cid > 0:
#                 current_pids = set( table.get_changed_df().index.tolist() )
#                 new_ids: Set[int] = label_map.get( cid, set() )
#                 deleted_pids = current_pids - new_ids
#                 added_pids = new_ids - current_pids
#                 nc = len(added_pids) + len(deleted_pids)
#                 if nc > 0:
#                     n_changes = n_changes + nc
#                     if n_changes:         lgm().log(f"\n TM----> update_selection[{cid}]" )
#                     if len(deleted_pids): lgm().log(f"    ######## deleted: {deleted_pids} ")
#                     if len(added_pids):   lgm().log(f"    ######## added: {added_pids} ")
#                     for polyId in added_pids: changed_pids[polyId] = cid
#                     for polyId in deleted_pids:
#                         if polyId not in  changed_pids.keys(): changed_pids[polyId] = 0
#                     table._remove_rows( deleted_pids )
#                     for polyId in added_pids:
#                         row = directory.df.loc[polyId].to_dict()
#                         row.update( dict( class=cid, Index=polyId ) )
# #                        lgm().log(f" TableManager.update_selection[{cid},{polyId}]: row = {row}")
#                         table._add_row( row.items() )
#         if n_changes > 0:
#             for (polyId,cid) in changed_pids.items():
#                 directory.edit_cell( polyId, "cid", cid )
# #        self._broadcast_selection_events = True
#
#         # directory_table = self._tables[0]
#         # for (cid, pids) in label_map.items():
#         #     table = self._tables[cid]
#         #     if cid > 0:
#         #         for polyId in pids:
#         #             directory_table.edit_cell( polyId, "cid", cid )
#         #             self._class_map[polyId] = cid
#         #             row = directory_table.df.loc[polyId]
#         #             table.add_row( row )
#         #
#         #         index_list: List[int] = selection_table.index.tolist()
#         #         table.edit_cell( index_list, "cid", cid )
#         #         lgm().log( f" Edit directory table: set _classes for gindices {index_list} to {cid}")
#         #         table.df = pd.concat( [table.df, selection_table] )
#         #         lgm().log(f" Edit class table[{cid}]: add pids {pids}, append selection_table with shape {selection_table.shape}")
#
#     def mark_selection(self):
#         from spectraclass.model.labels import LabelsManager, lm
#         selection_table: pd.DataFrame = self._tables[0].df.loc[self._current_selection]
#         cid: int = lm().mark_points( selection_table.index.to_numpy(np.int32) )
#         self._class_map[self._current_selection] = cid
#         for table_index, table in enumerate( self._tables ):
#             if table_index == 0:
#                 index_list: List[int] = selection_table.index.tolist()
#                 lgm().log( f" -----> Setting cid[{cid}] for gindices[:10]= {index_list[:10]}, current_selection = {self._current_selection}, class map nonzero = {np.count_nonzero(self._class_map)}")
#                 table.edit_cell( index_list, "cid", cid )
#             else:
#                 if table_index == cid:    table.df = pd.concat( [ table.df, selection_table ] ).drop_duplicates()
#                 else:                     self.drop_rows( table_index, self._current_selection )

    @property
    def selected_class(self):
        return int( self._wTablesWidget.selected_index )

    @property
    def selected_table(self):
        return self._tables[ self.selected_class ]

    def _handle_table_event(self, event, widget):
        lgm().log( f"\n  TABLE: handle_event: {event}" )
        ename = event['name']
        if( ename == 'sort_changed'):
            cname = event['new']['column']
            lgm().log(f"  sort_changed: {ename}[{cname}]: {self._cols}")
            self._current_column_index = self._cols.index( cname )
            lgm().log(f"  ... col-sel ---> ci={self._current_column_index}")
            self._clear_selection()
        elif (ename == 'selection_changed'):
            if (event['source'] == 'gui'):   #  and self._broadcast_selection_events:
                rows = event["new"]
                if len( rows ) == 1 or self.is_block_selection(event):
                    df = self.selected_table.get_changed_df()
                    new_selection = df.index[ rows ].to_list()
                    lgm().log(f"  **TABLE-> new selection event: rows = {rows}, indices = {new_selection}")
                    if new_selection != self._current_selection:
                        self._current_selection = new_selection
                        self.broadcast_selection_event( self._current_selection )

    def is_block_selection( self, event: Dict ) -> bool:
        old, new = event['old'], event['new']
        lgm().log(   f"   **TABLE->  is_block_selection: old = {old}, new = {new}"  )
        if (len(old) == 1) and (new[-1] == old[ 0]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        if (len(old) >  1) and (new[-1] == old[-1]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        return False

    def broadcast_selection_event(self, pids: List[int] ):
        from spectraclass.application.controller import app
        from spectraclass.model.labels import lm
        from spectraclass.gui.spatial.widgets.markers import Marker
# if self._broadcast_selection_events:
        item_str = "" if len(pids) > 8 else f",  pids={pids}"
        lgm().log( f" **TABLE-> gui.selection_changed, nitems={len(pids)}{item_str}" )
        cid = lm().current_cid if self.mark_on_selection else 0
        app().add_marker( "table", Marker( pids, cid ) )

    def _createTable( self, tab_index: int ) -> qgrid.QgridWidget:
        assert self._dataFrame is not None, " TableManager has not been initialized "
        col_opts = dict( editable=False ) #
        grid_opts = dict(  editable=False, maxVisibleRows=40 )
        if tab_index == 0:
            data_table = self._dataFrame.sort_values(self._cols[0] )
            data_table.insert( len(self._cols)-1, "cid", 0, True )
            wTable = qgrid.show_grid( data_table, column_options=col_opts, grid_options=grid_opts, show_toolbar=False )
        else:
            empty_catalog = {col: np.empty( [0], 'U' ) for col in self._cols}
            dFrame: pd.DataFrame = pd.DataFrame(empty_catalog, dtype='U', index=pd.Int64Index( [], name="Index" ) )
            wTable = qgrid.show_grid( dFrame, column_options=col_opts, grid_options=grid_opts, show_toolbar=False )
        wTable.on( tl.All, self._handle_table_event )
        wTable.layout = ipw.Layout( width="auto", height="100%", max_height="1000px" )
        return wTable

    def _createGui( self ) -> ipw.VBox:
        wSelectionPanel = self._createSelectionPanel()
        self._wTablesWidget = self._createTableTabs()
        return ipw.VBox([wSelectionPanel, self._wTablesWidget])

    def _createSelectionPanel( self ) -> ipw.HBox:
        self._wFind = ipw.Text( value='', placeholder='Filter table rows', description='Filter:', disabled=False, continuous_update = False, tooltip="Filter selected column with regex" )
        self._wFind.observe(self._process_filter, 'value')
        wFindOptions = self._createFindOptionButtons()
        wSelectionPanel = ipw.HBox( [ self._wFind, wFindOptions ] )
        wSelectionPanel.layout = ipw.Layout( justify_content = "center", align_items="center", width = "auto", height = "50px", min_height = "50px", border_width=1, border_color="white" )
        return wSelectionPanel

    def _createFindOptionButtons(self):
        if self._search_widgets is None:
            self._search_widgets = dict(
                find_select=     ToggleButton( [ 'search-location', 'th-list'], ['find','select'], [ 'find first', 'select all'] ),
                case_sensitive=  ToggleButton( ['font', 'asterisk'], ['true', 'false'],['case sensitive', 'case insensitive']),
                match=           ToggleButton( ['caret-square-left', 'caret-square-right', 'caret-square-down'], ['begins-with', 'ends-with', 'contains'], ['begins with', 'ends with', 'contains'])
            )
            for name, widget in self._search_widgets.items():
                widget.add_listener( partial( self._process_find_options, name ) )
                self._match_options[ name ] = widget.state

        buttonbox =  ipw.HBox( [ w.gui() for w in self._search_widgets.values() ] )
        buttonbox.layout = ipw.Layout( width = "300px", min_width = "300px", height = "auto" )
        return buttonbox

    def _process_filter(self, event: Dict[str, str]):
        match = self._match_options['match']
        case_sensitive = ( self._match_options['case_sensitive'] == "true" )
        df: pd.DataFrame = self.selected_table.get_changed_df()
        cname = self._cols[ self._current_column_index ]
        np_coldata = df[cname].to_numpy( dtype='U' )
        if not case_sensitive: np_coldata = np.char.lower( np_coldata )
        match_str = event['new'] if case_sensitive else event['new'].lower()
        if match == "begins-with":   mask = np.char.startswith( np_coldata, match_str )
        elif match == "ends-with":   mask = np.char.endswith( np_coldata, match_str )
        elif match == "contains":    mask = ( np.char.find( np_coldata, match_str ) >= 0 )
        else: raise Exception( f"Unrecognized match option: {match}")
        lgm().log( f" **TABLE-> process_find[ M:{match} CS:{case_sensitive} col:{self._current_column_index} ], coldata shape = {np_coldata.shape}, match_str={match_str}, coldata[:10]={np_coldata[:10]}" )
        self._current_selection = df.index[mask].to_list()
        lgm().log(f"  **TABLE->  cname = {cname}, mask shape = {mask.shape}, mask #nonzero = {np.count_nonzero(mask)}, #selected = {len(self._current_selection)}, selection[:8] = {self._current_selection[:8]}")
        self._select_find_results( )

    def _clear_selection(self):
        self._current_selection = []
        self._wFind.value = ""

    def _select_find_results(self ):
        if len( self._wFind.value ) > 0:
            find_select = self._match_options['find_select']
            selection = self._current_selection if find_select=="select" else self._current_selection[:1]
            lgm().log(f" **TABLE-> apply_selection[ {find_select} ], nitems: {len(selection)}")
            self.selected_table.change_selection( selection )
            self.broadcast_selection_event( selection )

    def _process_find_options(self, name: str, state: str ):
        self._match_options[ name ] = state

    def _createTableTabs(self) -> ipw.Tab:
        wTab = ipw.Tab()
        self._tables.append( self._createTable( 0 ))
        for iC, ctitle in enumerate( LabelsManager.instance().labels[1:], 1 ):
            self._tables.append(  self._createTable( iC ) )
        wTab.children = self._tables
        wTab.set_title( 0, 'Catalog')
        for iC, ctitle in enumerate( LabelsManager.instance().labels[1:], 1 ):
            wTab.set_title( iC, ctitle )
        return wTab

    def refresh(self):
        self._wGui = None
        self._tables: List[qgrid.QgridWidget] = []

    def _handle_key_event(self, event: Dict ):
        lgm().log( f" ################## handle_key_event: {event}  ################## ################## ##################" )

    def gui( self, **kwargs ) -> ipw.VBox:
        if self._wGui is None:
            self.init( **kwargs )
            self._wGui = self._createGui()
            self._wGui.layout = ipw.Layout(width='auto', flex='1 0 500px')
        return self._wGui

