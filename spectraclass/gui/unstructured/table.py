import qgrid, logging
from typing import List, Union, Tuple, Optional, Dict, Callable, Set
from spectraclass.util.logs import LogManager, lgm, exception_handled
from functools import partial
import xarray as xa
import numpy as np
import pandas as pd
import ipywidgets as ipw
from spectraclass.gui.widgets import ToggleButton
from spectraclass.data.base import DataManager, dm
from spectraclass.gui.points import PointCloudManager
from traitlets import traitlets
from spectraclass.model.labels import LabelsManager
import traitlets.config as tlc
from spectraclass.model.base import SCSingletonConfigurable

def tm(): return TableManager.instance()

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

    def init(self, **kwargs):
        catalog: Dict[str,np.ndarray] = kwargs.get( 'catalog', None )
        project_data: xa.Dataset = dm().loadCurrentProject("table")
        if catalog is None:  catalog = { tcol: project_data[tcol].values for tcol in dm().modal.METAVARS }
        nrows = catalog[ dm().modal.METAVARS[0] ].shape[0]
        lgm().log( f"Catalog: nrows = {nrows}, entries: {[ f'{k}:{v.shape}' for (k,v) in catalog.items() ]}" )
        self._dataFrame: pd.DataFrame = pd.DataFrame( catalog, dtype='U', index=pd.Int64Index( range(nrows), name="Index" ) )
        self._cols = list(catalog.keys()) + [ "Class" ]
        self._class_map = np.zeros( nrows, np.int32 )
        self._flow_class_map = np.zeros( nrows, np.int32 )

    def clear_tables(self):
        nrows = self._dataFrame.shape[0]
        self._class_map = np.zeros(nrows, np.int32)
        for cid, table in enumerate(self._tables):
            if cid == 0:
                table.df["Class"].values[:] = 0
            else:
                table.df = table.df.iloc[0:0]

    def update_selection(self):
        from spectraclass.model.labels import LabelsManager, lm
        self.clear_tables()
        label_map: Dict[int,Set[int]] = lm().getLabelMap()
        lgm().log(f" update_selection: set label_map = {label_map}")
        directory_table = self._tables[0]
        for (cid, pids) in label_map.items():
            table = self._tables[cid]
            if cid > 0:
                for pid in pids:
                    directory_table.edit_cell( pid, "Class", cid )
                    self._class_map[pid] = cid
                    row = directory_table.df.loc[pid]
                    table.add_row( row )

                index_list: List[int] = selection_table.index.tolist()
                table.edit_cell( index_list, "Class", cid )
                lgm().log( f" Edit directory table: set classes for indices {index_list} to {cid}")
                table.df = pd.concat( [table.df, selection_table] )
                lgm().log(f" Edit class table[{cid}]: add pids {pids}, append selection_table with shape {selection_table.shape}")

    def mark_selection(self):
        from spectraclass.model.labels import LabelsManager, lm
        selection_table: pd.DataFrame = self._tables[0].df.loc[self._current_selection]
        cid: int = lm().mark_points( selection_table.index.to_numpy(np.int32) )
        self._class_map[self._current_selection] = cid
        for table_index, table in enumerate( self._tables ):
            if table_index == 0:
                index_list: List[int] = selection_table.index.tolist()
                lgm().log( f" -----> Setting cid[{cid}] for indices[:10]= {index_list[:10]}, current_selection = {self._current_selection}, class map nonzero = {np.count_nonzero(self._class_map)}")
                table.edit_cell( index_list, "Class", cid )
            else:
                if table_index == cid:    table.df = pd.concat( [ table.df, selection_table ] ).drop_duplicates()
                else:                     self.drop_rows( table_index, self._current_selection )

    @property
    def selected_class(self):
        return int( self._wTablesWidget.selected_index )

    @property
    def selected_table(self):
        return self._tables[ self.selected_class ]

    def drop_rows(self, cid: int, pids: List[int]) :
        try: self._tables[cid].remove_rows(pids)
        except: pass
        self._tables[cid].df.drop(index=pids, inplace=True, errors="ignore" )
        lgm().log( f"TABLE[{cid}]: Dropping rows in class map: {pids}")

    def _handle_table_event(self, event, widget):
        ename = event['name']
        if( ename == 'sort_changed'):
            cname = event['new']['column']
            lgm().log(f"  handle_table_event: {ename}[{cname}]: {self._cols}")
            self._current_column_index = self._cols.index( cname )
            lgm().log(f"  ... col-sel ---> ci={self._current_column_index}")
            self._clear_selection()
        elif (ename == 'selection_changed'):
            if event['source'] == 'gui':
                rows = event["new"]
                if len( rows ) == 1 or self.is_block_selection(event):
                    lgm().log( f" TABLE.row-sel --->  {rows}" )
                    df = self.selected_table.get_changed_df()
                    lgm().log( f" TABLE[{self.selected_class}].row-index[:10] --->  {df.index[:10].to_list()}")
                    self._current_selection = df.index[ rows ].to_list()
                    lgm().log( f" TABLE[{self.selected_class}].current_selection[:10] --->  {self._current_selection[:10]}")
                    self.broadcast_selection_event( self._current_selection )

    def is_block_selection( self, event: Dict ) -> bool:
        old, new = event['old'], event['new']
        lgm().log(   f"  ------->  is_block_selection: old = {old}, new = {new}"  )
        if (len(old) == 1) and (new[-1] == old[ 0]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        if (len(old) >  1) and (new[-1] == old[-1]) and ( len(new) == (new[-2]-new[-1]+1)): return True
        return False

    def is_block_selection1( self, event: Dict ) -> bool:
        row_list = event['new'].sort()
        return row_list == list( range( row_list[0], row_list[-1]+1 ) )

    def broadcast_selection_event(self, pids: List[int] ):
        from spectraclass.application.controller import app
        from spectraclass.model.labels import LabelsManager, lm
        from spectraclass.model.base import Marker
        item_str = "" if len(pids) > 8 else f",  pids={pids}"
        lgm().log(f"TABLE.gui->selection_changed, nitems={len(pids)}{item_str}")
        app().add_marker( Marker( pids, lm().current_cid ) )

    def _createTable( self, tab_index: int ) -> qgrid.QgridWidget:
        assert self._dataFrame is not None, " TableManager has not been initialized "
        col_opts = dict( editable=False ) #
        grid_opts = dict(  editable=False, maxVisibleRows=40 )
        if tab_index == 0:
            data_table = self._dataFrame.sort_values(self._cols[0] )
            data_table.insert( len(self._cols)-1, "Class", 0, True )
            wTable = qgrid.show_grid( data_table, column_options=col_opts, grid_options=grid_opts, show_toolbar=False )
        else:
            empty_catalog = {col: np.empty( [0], 'U' ) for col in self._cols}
            dFrame: pd.DataFrame = pd.DataFrame(empty_catalog, dtype='U', index=pd.Int64Index( [], name="Index" ) )
            wTable = qgrid.show_grid( dFrame, column_options=col_opts, grid_options=grid_opts, show_toolbar=False )
        wTable.on( traitlets.All, self._handle_table_event )
        wTable.layout = ipw.Layout( width="auto", height="100%", max_height="1000px" )
        return wTable

    def _createGui( self ) -> ipw.VBox:
        wSelectionPanel = self._createSelectionPanel()
        self._wTablesWidget = self._createTableTabs()
        return ipw.VBox([wSelectionPanel, self._wTablesWidget])

    def _createSelectionPanel( self ) -> ipw.HBox:
        self._wFind = ipw.Text( value='', placeholder='Find items', description='Find:', disabled=False, continuous_update = False, tooltip="Search in sorted column" )
        self._wFind.observe(self._process_find, 'value')
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

    def _process_find(self, event: Dict[str,str]):
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
        lgm().log( f"process_find[ M:{match} CS:{case_sensitive} col:{self._current_column_index} ], coldata shape = {np_coldata.shape}, match_str={match_str}, coldata[:10]={np_coldata[:10]}" )
        self._current_selection = df.index[mask].to_list()
        lgm().log(f" --> cname = {cname}, mask shape = {mask.shape}, mask #nonzero = {np.count_nonzero(mask)}, #selected = {len(self._current_selection)}, selection[:8] = {self._current_selection[:8]}")
        self._select_find_results( )

    def _clear_selection(self):
        self._current_selection = []
        self._wFind.value = ""

    def _select_find_results(self ):
        if len( self._wFind.value ) > 0:
            find_select = self._match_options['find_select']
            selection = self._current_selection if find_select=="select" else self._current_selection[:1]
            lgm().log(f"apply_selection[ {find_select} ], nitems: {len(selection)}")
            self.selected_table.change_selection( selection )
            self.broadcast_selection_event( selection )

    def _process_find_options(self, name: str, state: str ):
        lgm().log( f"process_find_options[{name}]: {state}" )
        self._match_options[ name ] = state
        self._process_find( dict( new=self._wFind.value ) )

    def _createTableTabs(self) -> ipw.Tab:
        wTab = ipw.Tab()
        self._tables.append( self._createTable( 0 ))
        wTab.set_title( 0, 'Catalog')
        for iC, ctitle in enumerate( LabelsManager.instance().labels[1:], 1 ):
            self._tables.append(  self._createTable( iC ) )
            wTab.set_title( iC, ctitle )
        wTab.children = self._tables
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

