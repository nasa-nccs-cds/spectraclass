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
from bokeh.core.property.container import ColumnData, Seq
from typing import List, Union, Tuple, Optional, Dict, Callable, Set, Any, Iterable
from enum import Enum

class bkSpreadsheet:

    def __init__(self, data: Union[pd.DataFrame,xa.DataArray], **kwargs ):
        self._current_page_data: pd.DataFrame = None
        self._rows_per_page = kwargs.get( 'rows_per_page', 100 )
        self._current_page = None
        self.current_action: str = None
        self._paging_enabled = True
        self._selection: np.ndarray = None
        self._dataFrame: pd.DataFrame = None
        self._source: ColumnDataSource = None
        if isinstance( data, pd.DataFrame ):
            self._dataFrame = data
        elif isinstance( data, xa.DataArray ):
            assert data.ndim == 2, f"Wrong DataArray.ndim for bkSpreadsheet ({data.ndim}): must have ndim = 2"
            self._dataFrame = data.to_pandas()
        else:
            raise TypeError( f"Unsupported data class supplied to bkSpreadsheet: {data.__class__}" )
        self._columns = [TableColumn(field=cid, title=cid, sortable=True) for cid in self._dataFrame.columns]
        self.current_page = kwargs.get('init_page', 0)
        self._selection = np.full( [ self._dataFrame.shape[0] ], False, np.bool )
        self._table = DataTable( source=self._source, columns=self._columns, width=400, height=280, selectable="checkbox", index_position=None )

    def pids(self, page = True ) -> np.ndarray:
        return self.idxs2pids( np.array( self._source.data["index"] ) ) if page else self._dataFrame.index.to_numpy()

    @property
    def current_page(self) -> int:
        return self._current_page

    @current_page.setter
    def current_page(self, page_index: int):
        if self._current_page != page_index:
            self._current_page = page_index
            self._paging_enabled = True
            self._current_page_data = self._dataFrame.iloc[ self.page_start:self.page_end ]
            if self._source is None:
                self._source = ColumnDataSource( self._current_page_data )
                self._source.selected.on_change("indices", self._process_selection_change )
            self._source.data = self._current_page_data
            lgm().log( f"\n   bkSpreadsheet[{page_index}]-> current_page: source.data = {self._source.data}" )
            self.update_selection()

    @exception_handled
    def update_selection(self):
        selection_indices: np.ndarray = np.nonzero( self._selection )[0]
        lgm().log(f" update_selection[{self._current_page}] -> selection_indices = {selection_indices.tolist()}")
        self.set_selection( selection_indices )

    def idxs2pids(self, idxs: np.ndarray) -> np.ndarray:
        if self._paging_enabled:
            return idxs + self.page_start
        else:
            page_pids: np.ndarray = self.pids()
            return page_pids[idxs]

    @property
    def page_start(self) -> int:
        return self._current_page * self._rows_per_page

    @property
    def page_end(self) -> int:
        return self.page_start + self._rows_per_page

    def pid2idx(self, pid: int ) -> int:
        if self._paging_enabled:
            idx = pid - self.page_start
            return idx if self.valid_idx( idx ) else -1
        else:
            try:
                page_pids = self.pids()
                return page_pids.tolist().index( pid )
            except ValueError:
                return -1

    def pids2idxs(self, pids: np.ndarray ) -> np.ndarray:
        if self._paging_enabled:
            idx: np.ndarray = pids - self.page_start
            return idx[ (idx>=0) & (idx<self._rows_per_page) ]
        else:
            page_pids: np.ndarray = self.pids()
            return np.nonzero( page_pids.isin( pids ) )

    def valid_idx(self, idx: int ):
        return (idx>=0) and (idx<self._rows_per_page)

    @property
    def page_data(self) -> pd.DataFrame:
        return self._current_page_data

    def to_df( self ) -> pd.DataFrame:
        return self._source.to_df()

    def _process_selection_change(self, attr: str, old: List[int], new: List[int] ):
        old_pids: np.ndarray = self.idxs2pids( np.array(old) )
        new_pids: np.ndarray = self.idxs2pids( np.array(new) )
        if len( old ): self._selection[ old_pids ] = False
        if len( new ): self._selection[ new_pids ] = True

    def selection_callback( self, callback: Callable[[np.ndarray,np.ndarray],None] ):
        self._source.selected.on_change("indices", partial( self._exec_selection_callback, callback ) )

    def _exec_selection_callback(self, callback: Callable[["bkSpreadsheet",np.ndarray,np.ndarray],None], attr, old, new ):
        old_ids, new_ids = np.array( old ), np.array( new )
        lgm().log( f"\n-----------> exec_selection_callback: old = {old}, new = {new}, old_ids ={old_ids}, new_ids ={new_ids}\n" )
        callback( self, self.idxs2pids( old_ids ), self.idxs2pids( new_ids ) )

    def set_selection(self, pids: np.ndarray ):
        idxs: List[int] = self.pids2idxs( pids ).tolist()
        lgm().log( f" set_selection[{self._current_page}] -> idxs = {idxs}, pids = {pids.tolist()}" )
        self._source.selected.indices = idxs
        self._current_action = None

    def set_col_data(self, colname: str, value: Any ):
        self._source.data[colname].fill( value )

    def get_col_data(self, colname: str ) -> List:
        return self._source.data[colname]

    def from_df(self, pdf: pd.DataFrame ):
#        data = self._source.from_df( pdf )
        self._source.data = pdf
        self._paging_enabled = False
        lgm().log( f"\n   bkSpreadsheet-> Update page, pdf cols = {pdf.columns}, pdf shape = {pdf.shape}, source.data = {self._source.data}, col data = {self._source.data['cid']}")

    def patch_data_element(self, colname: str, pid: int, value ):
        idx = self.pid2idx( pid )
        if idx >= 0:
            self._source.patch( { colname: [( slice(idx,idx+1), [value] )] } )
            self._dataFrame.at[ pid, colname ] = value

    def get_selection( self ) -> List[int]:
        return self.idxs2pids(self._source.selected.indices)

    def gui(self) -> BokehModel:
        return BokehModel(self._table)

    def page_widget(self):
        nRows = self._dataFrame.shape[0]
        npages = math.ceil( nRows/self._rows_per_page )
        widget = ipw.Dropdown( options=list(range(npages)), description = "Page", index=0 )
        widget.observe( self._update_page, "value" )
        return widget

    def _update_page(self, event: Dict[str,str] ):
        if event['type'] == 'change':
            self.current_action = "page"
            self.current_page = event['new']

class TableManager(SCSingletonConfigurable):

    def __init__(self):
        super(TableManager, self).__init__()
        self._wGui: ipw.VBox = None
        self._dataFrame: pd.DataFrame = None
        self._tables: List[bkSpreadsheet] = []
        self._cols: List[str] = None
        self._wTablesWidget: ipw.Tab = None
        self._current_column_index: int = 0
        self._current_selection: np.ndarray = None
        self._search_widgets: Dict[str,ToggleButton] = None
        self._match_options = {}
        self._events = []
        self._broadcast_selection_events = True
        self.mark_on_selection = False
        self.ignorable_actions = ["page"]

    def init(self, **kwargs):
        catalog: Dict[str,np.ndarray] = kwargs.get( 'catalog', None )
        project_data: xa.Dataset = dm().loadCurrentProject("table")
        if catalog is None:  catalog = { tcol: project_data[tcol].values for tcol in dm().modal.METAVARS }
        nrows = catalog[ dm().modal.METAVARS[0] ].shape[0]
        lgm().log( f"Catalog: nrows = {nrows}, entries: {[ f'{k}:{v.shape}' for (k,v) in catalog.items() ]}" )
        self._dataFrame: pd.DataFrame = pd.DataFrame( catalog, dtype='U', index=pd.Int64Index( range(nrows), name="index" ) )
        self._cols = list(catalog.keys())
        self._dataFrame.insert(len(self._cols), "cid", 0, True)
        lgm().log(f"\nDataFrame: cols = {self._dataFrame.columns}, catalog cols = {self._cols}\n" )

    def edit_table(self, cid: int, pids: np.ndarray, column: str, value: Any ):
         table: bkSpreadsheet = self._tables[cid]
         for pid in pids.tolist():
            table.patch_data_element( column, pid, value )

    def  clear_table(self,cid: int):
        table: bkSpreadsheet = self._tables[cid]
        table.set_col_data( "cid", 0 )


    def update_selection(self, action: str ):
        from spectraclass.model.labels import LabelsManager, lm
        label_map: Dict[int,Set[int]] = lm().getLabelMap( True )
        table: bkSpreadsheet = None
        lgm().log(f"\n TM----> update_selection:")
        for (cid,table) in enumerate(self._tables):
            table.current_action = action
            if cid > 0:
                current_pids: Set[int] = set( table.pids(False).tolist() )
                new_ids: Set[int] = label_map.get( cid, set() )
                deleted_pids = current_pids - new_ids
                added_pids = new_ids - current_pids
                nc = len(added_pids) + len(deleted_pids)
                if nc > 0:
                    lgm().log(f" TM----> UPDATE CLASS TABLE [class={cid}]:" )
                    if len(deleted_pids): lgm().log(f"    ######## deleted pids= {deleted_pids} ")
                    if len(added_pids):   lgm().log(f"    ######## added pids= {added_pids} ")
                    cid_mask: np.ndarray = self._dataFrame.index.isin( new_ids )
                    lgm().log( f" TM----> dataFrame.index = {self._dataFrame.index}, current_pids={current_pids}, cid_mask = {cid_mask}")
                    df: pd.DataFrame = self._dataFrame[ cid_mask ].assign( cid=cid )
                    lgm().log(f" TM----> Add rows to class[{cid}], df.shape = {df.shape}, mask shape = {cid_mask.shape}")
                    table.from_df( df )

    @property
    def selected_class(self) -> int:
        return int( self._wTablesWidget.selected_index )

    @property
    def selected_table(self) -> bkSpreadsheet:
        return self._tables[ self.selected_class ]

    def _handle_table_selection(self, table: bkSpreadsheet, old: np.ndarray, new: np.ndarray ):
        lgm().log(f"  **TABLE-> new selection event, indices:  {old} -> {new}")
        action: str = table.current_action
        if action not in self.ignorable_actions:
            self._current_selection = new
            self.broadcast_selection_event( new )

    def is_block_selection( self, old: List[int], new: List[int] ) -> bool:
        lgm().log(   f"   **TABLE->  is_block_selection: old = {old}, new = {new}"  )
        if (len(old) == 1) and (new[-1] == old[ 0]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        if (len(old) >  1) and (new[-1] == old[-1]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        return False

    def broadcast_selection_event(self, pids: np.ndarray ):
        from spectraclass.application.controller import app
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.model.base import Marker
        item_str = "" if pids.size > 8 else f",  pids={pids}"
        lgm().log(f" **TABLE-> gui.selection_changed, nitems={pids.size}{item_str}")
        cid = lm().current_cid if self.mark_on_selection else 0
        app().add_marker( "table", Marker( pids, cid ) )

    def _createTable( self, tab_index: int ) -> bkSpreadsheet:
        assert self._dataFrame is not None, " TableManager has not been initialized "
        if tab_index == 0:
            bkTable = bkSpreadsheet( self._dataFrame )
        else:
            empty_catalog = {col: np.empty( [0], 'U' ) for col in self._cols}
            dFrame: pd.DataFrame = pd.DataFrame(empty_catalog, dtype='U' )
            bkTable = bkSpreadsheet( dFrame )
        bkTable.selection_callback(self._handle_table_selection)
        return bkTable

    def _createGui( self ) -> ipw.VBox:
        self._wTablesWidget = self._createTableTabs()
        wSelectionPanel = self._createSelectionPanel()
        return ipw.VBox([wSelectionPanel, self._wTablesWidget])

    def _createSelectionPanel( self ) -> ipw.HBox:
        self._wFind = ipw.Text( value='', placeholder='Find items', description='Find:', disabled=False, continuous_update = False, tooltip="Search in sorted column" )
        self._wFind.observe(self._process_find, 'value')
        wFindOptions = self._createFindOptionButtons()
        wPages = self._tables[0].page_widget()
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
        self._current_selection = df.index[mask].to_numpy()
        lgm().log(f"  **TABLE->  cname = {cname}, mask shape = {mask.shape}, mask #nonzero = {np.count_nonzero(mask)}, #selected = {len(self._current_selection)}, selection[:8] = {self._current_selection[:8]}")
        self._select_find_results( )

    def _clear_selection(self):
        self._current_selection = None
        self._wFind.value = ""

    def _select_find_results(self ):
        if len( self._wFind.value ) > 0:
            find_select = self._match_options['find_select']
            selection: np.ndarray = self._current_selection if find_select=="select" else self._current_selection[:1]
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
        self._current_page_data = None
        self._current_page = 0

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
        #             directory_table.edit_cell( pid, "cid", cid )
        #             self._class_map[pid] = cid
        #             row = directory_table.df.loc[pid]
        #             table.add_row( row )
        #
        #         index_list: List[int] = selection_table.index.tolist()
        #         table.edit_cell( index_list, "cid", cid )
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
#                 directory.edit_cell( pid, "cid", cid )
# #        self._broadcast_selection_events = True
#
#         # directory_table = self._tables[0]
#         # for (cid, pids) in label_map.items():
#         #     table = self._tables[cid]
#         #     if cid > 0:
#         #         for pid in pids:
#         #             directory_table.edit_cell( pid, "cid", cid )
#         #             self._class_map[pid] = cid
#         #             row = directory_table.df.loc[pid]
#         #             table.add_row( row )
#         #
#         #         index_list: List[int] = selection_table.index.tolist()
#         #         table.edit_cell( index_list, "cid", cid )
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
#                 table.edit_cell( index_list, "cid", cid )
#             else:
#                 if table_index == cid:    table.df = pd.concat( [ table.df, selection_table ] ).drop_duplicates()
#                 else:                     self.drop_rows( table_index, self._current_selection )
