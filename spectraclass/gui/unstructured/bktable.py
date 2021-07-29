import logging, traceback
from spectraclass.util.logs import LogManager, lgm, exception_handled
from functools import partial
import numpy as np
import pandas as pd
import ipywidgets as ipw
from spectraclass.gui.widgets import ToggleButton
from bokeh.models.widgets import TextInput
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
        self._current_page = None
        self._dataFrame: pd.DataFrame = None
        self._filteredData: pd.DataFrame = None
        self._source: ColumnDataSource = None
        if isinstance( data, pd.DataFrame ):
            self._dataFrame = data
        elif isinstance( data, xa.DataArray ):
            assert data.ndim == 2, f"Wrong DataArray.ndim for bkSpreadsheet ({data.ndim}): must have ndim = 2"
            self._dataFrame = data.to_pandas()
        else:
            raise TypeError( f"Unsupported data class supplied to bkSpreadsheet: {data.__class__}" )
        self._columns = [TableColumn(field=cid, title=cid, sortable=True) for cid in self._dataFrame.columns]
        self._selection = np.full( [ self._dataFrame.shape[0] ], False, np.bool )
        self.current_page = kwargs.get('init_page', 0)
        self._table = DataTable( source=self._source, columns=self._columns, width=400, height=280, selectable="checkbox", index_position=None )

    @property
    def table_data(self) -> pd.DataFrame:
        return self._dataFrame if self._filteredData is None else self._filteredData.filter( items=self._dataFrame.columns )

    def clear_filter(self):
        self._filteredData = None
        self.refresh_page_data()

    def set_filter_data(self, filter_data: pd.DataFrame):
        self._current_page = 0
        self._filteredData = filter_data
        self.refresh_page_data()

    def pids(self) -> np.ndarray:
        return self.table_data.index.to_numpy()

    def page_pids(self) -> np.ndarray:
        return self.idxs2pids( np.array( self._source.data["index"] ) )

    @property
    def current_page(self) -> int:
        return self._current_page

    @current_page.setter
    def current_page(self, page_index: int):
        if (self._current_page != page_index) and (page_index is not None):
            self._current_page = page_index
            self.refresh_page_data()

    def refresh_page_data(self):
        self._current_page_data = self.table_data.iloc[ self.page_start:self.page_end ]
        if self._source is None:
            self._source = ColumnDataSource( self._current_page_data )
            self._source.selected.on_change("indices", self._process_selection_change )
        self._source.data = self._current_page_data
        self.refresh_selection()

    @exception_handled
    def refresh_selection(self):
        selection_indices: np.ndarray = self.pids2idxs( np.nonzero( self._selection )[0] )
        lgm().log(f" update_selection[{self._current_page}] -> selection_indices = {selection_indices.tolist()}")
        self._source.selected.indices = selection_indices.tolist()

    def idxs2pids(self, idxs: Union[List,np.ndarray]) -> np.ndarray:
        if len(idxs) == 0:
            return np.array([],dtype=np.int)
        else:
            page_idxs = idxs if isinstance(idxs, np.ndarray) else np.array(idxs)
            global_idxs = page_idxs + self.page_start
            global_pids: np.ndarray = self.pids()
            lgm().log(f" idxs2pids[{self._current_page}]: global_pids = {global_pids}, global_idxs = {global_idxs}")
            return global_pids[global_idxs]

    @property
    def page_start(self) -> int:
        return self._current_page * TableManager.rows_per_page

    @property
    def page_end(self) -> int:
        return self.page_start + TableManager.rows_per_page

    def pid2idx(self, pid: int ) -> int:
        try:
            pids = self.pids()
            idx = pids.tolist().index( pid ) - self.page_start
        except ValueError:
            return -1
        return idx if self.valid_idx(idx) else -1

    def pids2idxs(self, pids: Union[List,np.ndarray] ) -> np.ndarray:
        if len(pids) == 0:
            return np.array([],dtype=np.int)
        else:
            _pids: np.ndarray = pids if isinstance(pids, np.ndarray) else np.array(pids)
            global_idxs: np.ndarray = np.nonzero( np.isin( self.pids(), _pids ) )[0]
            idx: np.ndarray = global_idxs - self.page_start
            return idx[(idx >= 0) & (idx < TableManager.rows_per_page)]

    def valid_idx(self, idx: int ):
        return (idx>=0) and (idx<TableManager.rows_per_page)

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

    def _exec_selection_callback(self, callback: Callable[[np.ndarray,np.ndarray],None], attr, old, new ):
        old_ids, new_ids = np.array( old ), np.array( new )
        lgm().log( f"\n-----------> exec_selection_callback: old = {old}, new = {new}, old_ids ={old_ids}, new_ids ={new_ids}\n" )
        callback( self.idxs2pids( old_ids ), self.idxs2pids( new_ids ) )

    def set_selection(self, pids: np.ndarray, refresh: bool ):
        idxs: List[int] = self.pids2idxs( pids ).tolist()
        lgm().log( f" set_selection[{self._current_page}] -> idxs = {idxs}, pids = {pids.tolist()}" )
        if refresh: self._selection.fill( False )
        self._selection[ pids ] = True
        self.refresh_selection()

    def set_col_data(self, colname: str, value: Any ):
        self._source.data[colname].fill( value )

    def get_col_data(self, colname: str ) -> List:
        return self._source.data[colname]

    def from_df(self, pdf: pd.DataFrame ):
        self._dataFrame = pdf
        self.refresh_page_data()
        lgm().log( f"\n   bkSpreadsheet-> Update page, pdf cols = {pdf.columns}, pdf shape = {pdf.shape}, source.data = {self._source.data}, col data = {self._source.data['cid']}")

    def patch_data_element(self, colname: str, pid: int, value ):
        idx = self.pid2idx( pid )
        if idx >= 0:
            self._source.patch( { colname: [( slice(idx,idx+1), [value] )] } )
            self._dataFrame.at[ pid, colname ] = value
            if self._filteredData is not None:
                self._filteredData.at[pid, colname] = value

    def get_selection( self ) -> np.ndarray:
        return self.idxs2pids(self._source.selected.indices)

    def gui(self) -> BokehModel:
        return BokehModel(self._table)

class TableManager(SCSingletonConfigurable):
    rows_per_page = 100

    def __init__(self):
        super(TableManager, self).__init__()
        self._wGui: ipw.VBox = None
        self._dataFrame: pd.DataFrame = None
        self._tables: List[bkSpreadsheet] = []
        self._cols: List[str] = None
        self._wTablesWidget: ipw.Tab = None
        self._wPages: ipw.Dropdown = None
        self._current_column_index: int = 0
        self._current_selection: np.ndarray = None
        self._filter_state_widgets: Dict[str, ToggleButton] = None
        self._filter_column: ipw.Dropdown = None
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

    @property
    def selected_table_index(self) -> int:
        return int( self._wTablesWidget.selected_index )

    @property
    def selected_table(self) -> bkSpreadsheet:
        return self._tables[ self.selected_table_index ]

    def mark_points(self):
        from spectraclass.model.labels import LabelsManager, lm
        dtable = self._tables[ 0 ]
        pids: np.ndarray = dtable.get_selection()
        self.edit_table(0, pids, "cid", lm().current_cid )
        new_pids = set( pids.tolist() )
        for (cid,table) in enumerate(self._tables):
            if cid > 0:
                current_pids: Set[int] = set( table.pids().tolist() )
                if cid == lm().current_cid:
                    updated_pids = current_pids.union(new_pids)
                else:
                    updated_pids = current_pids - new_pids
                if len( updated_pids ) != len( current_pids ):
                    cid_mask: np.ndarray = self._dataFrame.index.isin(updated_pids)
                    df: pd.DataFrame = self._dataFrame[cid_mask].assign(cid=cid)
                    table.from_df(df)
                    lgm().log(f"  TABLE[{cid}] update: {current_pids} -> {updated_pids}" )

    def _handle_table_selection(self, old: np.ndarray, new: np.ndarray ):
        lgm().log(f"  **TABLE-> new selection event, indices:  {old} -> {new}")
        self.broadcast_selection_event( new )

    def is_block_selection( self, old: List[int], new: List[int] ) -> bool:
        lgm().log(   f"   **TABLE->  is_block_selection: old = {old}, new = {new}"  )
        if (len(old) == 1) and (new[-1] == old[ 0]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        if (len(old) >  1) and (new[-1] == old[-1]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        return False

    def broadcast_selection_event(self, pids: np.ndarray ):
        from spectraclass.application.controller import app
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.gui.plot import GraphPlotManager, gpm
        from spectraclass.model.base import Marker
        item_str = "" if pids.size > 8 else f",  pids={pids}"
        lgm().log(f" **TABLE-> gui.selection_changed, nitems={pids.size}{item_str}")
        gpm().plot_graph(pids)

    def _createTable( self, tab_index: int ) -> bkSpreadsheet:
        assert self._dataFrame is not None, " TableManager has not been initialized "
        if tab_index == 0:
            bkTable = bkSpreadsheet( self._dataFrame ) # , sequential=True )
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
        self._wFilter = ipw.Text(value='', placeholder='Filter rows', description='Filter:', disabled=False, continuous_update = False, tooltip="Filter selected column by regex")
        self._wFilter.observe(self._process_filter, 'value')
        wFindOptions = self._createFilterOptionButtons()
        self._wPages = self.page_widget()
        wSelectionPanel = ipw.HBox([self._wFilter, wFindOptions, self._wPages])
        wSelectionPanel.layout = ipw.Layout( justify_content = "center", align_items="center", width = "auto", height = "50px", min_height = "50px", border_width=1, border_color="white" )
        return wSelectionPanel

    def page_widget(self):
        nRows = self._dataFrame.shape[0]
        npages = math.ceil( nRows/self.rows_per_page )
        widget = ipw.Dropdown( options=list(range(npages)), description = "Page", index=0 )
        widget.observe( self._update_page, "value" )
        return widget

    @exception_handled
    def _update_page_widget(self):
        nRows: int = self.selected_table.table_data.shape[0]
        npages: int = math.ceil( nRows / self.rows_per_page )
        self._wPages.options = list(range(npages))
        self._wPages.index = 0
        lgm().log( f"  Update pages: npages = {npages}")

    def _update_page(self, event: Dict[str,str] ):
        if event['type'] == 'change':
            self.selected_table.current_page = event['new']

    def _createFilterOptionButtons(self):
        if self._filter_state_widgets is None:
            self._filter_state_widgets = dict(
                clear=           ToggleButton( ['redo'], ['reset'], ['clear/reset'] ),  #  "times-circle"?
                case_sensitive=  ToggleButton( ['font', 'asterisk'], ['true', 'false'],['case sensitive', 'case insensitive']),
                match=           ToggleButton( ['caret-square-left', 'caret-square-down', 'caret-square-right',  'caret-square-up'], ['begins-with', 'regex', 'ends-with', 'contains'], ['begins with', 'regex', 'ends with', 'contains']),
            )
            for name, widget in self._filter_state_widgets.items():
                widget.add_listener(partial(self._process_filter_options, name))
                self._match_options[ name ] = widget.state
            self._filter_column = ipw.Dropdown(options=self._dataFrame.columns, value=self._dataFrame.columns[0], description_tooltip='column', disabled=False)
            self._filter_column.layout = ipw.Layout(width ="120px", min_width ="120px", height ="27px")

        buttonbox =  ipw.HBox([w.gui() for w in self._filter_state_widgets.values()] + [self._filter_column])
        buttonbox.layout = ipw.Layout( width = "300px", min_width = "300px", height = "auto" )
        return buttonbox

    def _process_filter(self, event: Dict[str, str]):
        match = self._match_options['match']
        case_sensitive = ( self._match_options['case_sensitive'] == "true" )
        df: pd.DataFrame = self.selected_table.to_df()
        cname = self._filter_column.value
        np_coldata = df[cname]
        if not case_sensitive: np_coldata = np.char.lower( np_coldata )
        match_str = event['new'] if case_sensitive else event['new'].lower()
        if len( match_str ) == 0:
            self.selected_table.clear_filter()
        else:
            if match == "begins-with":   mask = np_coldata.str.startswith( match_str )
            elif match == "ends-with":   mask = np_coldata.str.endswith( match_str )
            elif match == "contains":    mask = np_coldata.str.contains(  match_str )
            elif match == "regex":       mask = np_coldata.str.match( r'{}'.format(match_str) )
            else: raise Exception( f"Unrecognized match option: {match}")
            self.selected_table.set_filter_data( df[mask] )
        self._update_page_widget()

    def _clear_selection(self):
        self._current_selection = None
        self._wFilter.value = ""

    def _process_filter_options(self, name: str, state: str):
        self._match_options[ name ] = state
        if name == "clear":
            self.selected_table.clear_filter()
            self._wFilter.value = ""

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
