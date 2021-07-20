import logging, traceback
from spectraclass.util.logs import LogManager, lgm, exception_handled
from functools import partial
import numpy as np
import pandas as pd
import ipywidgets as ipw
from spectraclass.gui.widgets import ToggleButton
from spectraclass.data.base import DataManager, dm
from spectraclass.model.labels import LabelsManager
from spectraclass.model.base import SCSingletonConfigurable
from jupyter_bokeh.widgets import BokehModel
import math, xarray as xa
from bokeh.models import ColumnDataSource, DataTable, CustomJS, TableColumn
from bokeh.core.property.container import ColumnData
from typing import List, Union, Tuple, Optional, Dict, Callable, Set

class bkSpreadsheet:

    def __init__(self, data: Union[pd.DataFrame,xa.DataArray] ):
        self._pdf: pd.DataFrame = None
        if isinstance( data, pd.DataFrame ):
            self._pdf = data
        elif isinstance( data, xa.DataArray ):
            assert data.ndim == 2, f"Wrong DataArray.ndim for bkSpreadsheet ({data.ndim}): must have ndim = 2"
            self._pdf = data.to_pandas()
        else:
            raise TypeError( f"Unsupported data class supplied to bkSpreadsheet: {data.__class__}" )
#        self._pdf = self._pdf.iloc[0:250,:]
        cols = [ str(col) for col in self._pdf.columns if str(col).lower() != "index" ]
        self._source: ColumnDataSource = ColumnDataSource( self._pdf )
        self._columns = [ TableColumn(field=cid, title=cid, sortable=False) for cid in cols ]
        self._table = DataTable( source=self._source, columns=self._columns, width=400, height=280, selectable="checkbox", index_position=0, index_header="index" )

    def to_df(self) -> pd.DataFrame:
        return self._source.to_df()

    def selection_callback( self, callback: Callable[[str,str,str],None] ):  # callback( attr, old, new )
        self._source.selected.on_change("indices", callback)

    def set_selection(self, indices: List[int] ):
        self._source.selected.indices = indices

    def set_col_data(self, colname: str, data: List ):
        self._source.data[colname] = data

    def get_col_data(self, colname: str ) -> List:
        return self._source.data[colname]

    def from_df(self, pdf: pd.DataFrame ):
        data = { str(cname): cdata.array for cname, cdata in pdf.items() }
        data['index'] = pdf.index.to_numpy()
        self._source.data = data

    def patch_data_element(self, colname: str, index: int, value ):
        self._source.patch( { colname: [( slice(index,index+1), [value] )] } )

    def patch_data(self, colname: str, indices, values ):
        self._source.patch( { colname: [( indices, values )] } )

    def get_selection( self ) -> List[int]:
        return self._source.selected.indices

    def gui(self) -> BokehModel:
        return BokehModel(self._table)

class TableManager(SCSingletonConfigurable):

    def __init__(self):
        super(TableManager, self).__init__()
        self._wGui: ipw.VBox = None
        self._rows_per_page = 100
        self._current_page = 0
        self._dataFrame: pd.DataFrame = None
        self._tables: List[bkSpreadsheet] = []
        self._cols: List[str] = None
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
        self._dataFrame: pd.DataFrame = pd.DataFrame( catalog, dtype='U', index=pd.Int64Index( range(nrows), name="index" ) )
        lgm().log(f"DataFrame: cols = {self._dataFrame.columns.names}" )
        self._cols = list(catalog.keys()) + [ "class" ]
        self._class_map = np.zeros( nrows, np.int32 )
        self._flow_class_map = np.zeros( nrows, np.int32 )

    def edit_table(self, cid, index, column, value ):
         table: bkSpreadsheet = self._tables[cid]
         table.patch_data( column, index, value )

    def update_table(self, cid, fire_event = True):
        table: bkSpreadsheet = self._tables[cid]
 #       table._update_table( triggered_by='update_table', fire_data_change_event=fire_event )

    def update_selection(self):
        from spectraclass.model.labels import LabelsManager, lm
 #       self._broadcast_selection_events = False
        label_map: Dict[int,Set[int]] = lm().getLabelMap()
        directory: pd.DataFrame = self._tables[0].to_df()
        table: bkSpreadsheet = None
        changed_pids = dict()
        lgm().log(f"\n TM----> update_selection:")
        n_changes = 0
        for (cid,table) in enumerate(self._tables):
            if cid > 0:
                pdf: pd.DataFrame = table.to_df()
                lgm().log(f" TM----> TABLE [class={cid}, cols = {pdf.columns}]")
                current_pids = set( table.get_col_data("index") )
                new_ids: Set[int] = label_map.get( cid, set() )
                deleted_pids = current_pids - new_ids
                added_pids = new_ids - current_pids
                nc = len(added_pids) + len(deleted_pids)
                if nc > 0:
                    n_changes = n_changes + nc
                    lgm().log(f" TM----> UPDATE CLASS TABLE [class={cid}]:" )
                    if len(deleted_pids): lgm().log(f"    ######## deleted pids= {deleted_pids} ")
                    if len(added_pids):   lgm().log(f"    ######## added pids= {added_pids} ")
                    for pid in added_pids:    changed_pids[pid] = cid
                    for pid in deleted_pids:  changed_pids[pid] = 0
                    pdf.drop( index=deleted_pids, inplace= True )
                    for pid in added_pids:
                        drow = directory.loc[ (directory['index'] == pid) ]
                        drow["class"] = cid
                        lgm().log(f" TM----> ADD ROW [class={cid},pid={pid}]: row = \n{drow}")
                        pdf.append( drow, sort=True )
                    table.from_df( pdf )
        if n_changes > 0:
            dtable = self._tables[0]
            for (pid,cid) in changed_pids.items():
                dtable.patch_data_element( "class", pid, cid )
                lgm().log(f" TM----> UPDATE DIRECTORY [pid={pid}] -> class = {cid}")

    @property
    def selected_class(self) -> int:
        return int( self._wTablesWidget.selected_index )

    @property
    def selected_table(self) -> bkSpreadsheet:
        return self._tables[ self.selected_class ]

    def _handle_table_event(self, attr, old, new ):
        print( f" _handle_table_event: {attr} {old} {new} ")
        if (attr == 'indices'):
            new_selection = new
            lgm().log(f"  **TABLE-> new selection event: indices = {new}")
            if new_selection != self._current_selection:
                self._current_selection = new_selection
                self.broadcast_selection_event( self._current_selection )

    def is_block_selection( self, old: List[int], new: List[int] ) -> bool:
        lgm().log(   f"   **TABLE->  is_block_selection: old = {old}, new = {new}"  )
        if (len(old) == 1) and (new[-1] == old[ 0]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        if (len(old) >  1) and (new[-1] == old[-1]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        return False

    def broadcast_selection_event(self, pids: List[int] ):
        from spectraclass.application.controller import app
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.model.base import Marker
# if self._broadcast_selection_events:
        item_str = "" if len(pids) > 8 else f",  pids={pids}"
        lgm().log(f" **TABLE-> gui.selection_changed, nitems={len(pids)}{item_str}")
        cid = lm().current_cid if self.mark_on_selection else 0
        app().add_marker( "table", Marker( pids, cid ) )

    def _get_page_data(self) -> pd.DataFrame:
        page_start = self._current_page * self._rows_per_page
        data_table = self._dataFrame.iloc[page_start:page_start + self._rows_per_page]
        data_table.insert(len(self._cols) - 1, "class", 0, True)
        return data_table

    def _createTable( self, tab_index: int ) -> bkSpreadsheet:
        assert self._dataFrame is not None, " TableManager has not been initialized "
        if tab_index == 0:
            bkTable = bkSpreadsheet( self._get_page_data() )
        else:
            empty_catalog = {col: np.empty( [0], 'U' ) for col in self._cols}
            dFrame: pd.DataFrame = pd.DataFrame(empty_catalog, dtype='U' )
            bkTable = bkSpreadsheet( dFrame )
        bkTable.selection_callback( self._handle_table_event )
        return bkTable

    def _createGui( self ) -> ipw.VBox:
        wSelectionPanel = self._createSelectionPanel()
        self._wTablesWidget = self._createTableTabs()
        return ipw.VBox([wSelectionPanel, self._wTablesWidget])

    def _createPageWidget(self):
        nRows = self._dataFrame.shape[0]
        npages = math.ceil( nRows/self._rows_per_page )
        widget = ipw.Dropdown( options=list(range(npages)), description = "Page", index=0 )
        widget.observe( self._update_page, "value" )
        return widget

    def _update_page(self, event: Dict[str,str] ):
        if event['type'] == 'change':
            self._current_page = event['new']
            directory: bkSpreadsheet = self._tables[0]
            page_data: pd.DataFrame = self._get_page_data()
            lgm().log( f" ** update_page, data shape = {page_data.shape}, cols = {page_data.columns}, index = {page_data.index.to_numpy()}")
            directory.from_df( page_data )

    def _createSelectionPanel( self ) -> ipw.HBox:
        self._wFind = ipw.Text( value='', placeholder='Find items', description='Find:', disabled=False, continuous_update = False, tooltip="Search in sorted column" )
        self._wFind.observe(self._process_find, 'value')
        wFindOptions = self._createFindOptionButtons()
        wPages = self._createPageWidget()
        wSelectionPanel = ipw.HBox( [ self._wFind, wFindOptions, wPages ] )
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

    def _process_find(self, event: Dict[str,str]):
        match = self._match_options['match']
        case_sensitive = ( self._match_options['case_sensitive'] == "true" )
        df: pd.DataFrame = self.selected_table.to_df()
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
            self.selected_table.set_selection( selection )
            self.broadcast_selection_event( selection )

    def _process_find_options(self, name: str, state: str ):
        lgm().log( f" **TABLE-> process_find_options[{name}]: {state}" )
        self._match_options[ name ] = state
        self._process_find( dict( new=self._wFind.value ) )

    def _createTableTabs(self) -> ipw.Tab:
        wTab = ipw.Tab()
        self._tables.append( self._createTable( 0 ) )
        wTab.set_title( 0, 'Catalog')
        for iC, ctitle in enumerate( LabelsManager.instance().labels[1:], 1 ):
            self._tables.append(  self._createTable( iC ) )
            wTab.set_title( iC, ctitle )
        wTab.children = [ t.gui() for t in self._tables ]
        return wTab

    def refresh(self):
        self._wGui = None
        self._tables = []

    def _handle_key_event(self, event: Dict ):
        lgm().log( f" ################## handle_key_event: {event}  ################## ################## ##################" )

    def gui( self, **kwargs ) -> ipw.VBox:
        if self._wGui is None:
            self.init( **kwargs )
            self._wGui = self._createGui()
            self._wGui.layout = ipw.Layout(width='auto', flex='1 0 500px')
        return self._wGui

#        self._broadcast_selection_events = True

#        self.update_table( 0, False )

        # directory_table = self._tables[0]
        # for (cid, pids) in label_map.items():
        #     table = self._tables[cid]
        #     if cid > 0:
        #         for pid in pids:
        #             directory_table.edit_cell( pid, "class", cid )
        #             self._class_map[pid] = cid
        #             row = directory_table.df.loc[pid]
        #             table.add_row( row )
        #
        #         index_list: List[int] = selection_table.index.tolist()
        #         table.edit_cell( index_list, "class", cid )
        #         lgm().log( f" Edit directory table: set classes for indices {index_list} to {cid}")
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
#                     for pid in added_pids: changed_pids[pid] = cid
#                     for pid in deleted_pids:
#                         if pid not in  changed_pids.keys(): changed_pids[pid] = 0
#                     table._remove_rows( deleted_pids )
#                     for pid in added_pids:
#                         row = directory.df.loc[pid].to_dict()
#                         row.update( dict( class=cid, Index=pid ) )
# #                        lgm().log(f" TableManager.update_selection[{cid},{pid}]: row = {row}")
#                         table._add_row( row.items() )
#         if n_changes > 0:
#             for (pid,cid) in changed_pids.items():
#                 directory.edit_cell( pid, "class", cid )
# #        self._broadcast_selection_events = True
#
#         # directory_table = self._tables[0]
#         # for (cid, pids) in label_map.items():
#         #     table = self._tables[cid]
#         #     if cid > 0:
#         #         for pid in pids:
#         #             directory_table.edit_cell( pid, "class", cid )
#         #             self._class_map[pid] = cid
#         #             row = directory_table.df.loc[pid]
#         #             table.add_row( row )
#         #
#         #         index_list: List[int] = selection_table.index.tolist()
#         #         table.edit_cell( index_list, "class", cid )
#         #         lgm().log( f" Edit directory table: set classes for indices {index_list} to {cid}")
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
#                 lgm().log( f" -----> Setting cid[{cid}] for indices[:10]= {index_list[:10]}, current_selection = {self._current_selection}, class map nonzero = {np.count_nonzero(self._class_map)}")
#                 table.edit_cell( index_list, "class", cid )
#             else:
#                 if table_index == cid:    table.df = pd.concat( [ table.df, selection_table ] ).drop_duplicates()
#                 else:                     self.drop_rows( table_index, self._current_selection )
