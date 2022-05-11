from spectraclass.data.spatial.manager import SpatialDataManager
import xarray as xa
import numpy as np
import os, glob
from collections import OrderedDict
from typing import List, Union, Tuple, Optional, Dict
from functools import partial
from typing import Optional, Dict


class PrepareInputsDialog(PreferencesDialog):

    def __init__( self, input_vars: Optional[Dict] = None, subsample: int = None, scope: QSettings.Scope = QSettings.UserScope, **kwargs  ):
        self.inputs = {} if input_vars is None else [ input_vars['embedding'] ] +  input_vars['directory'] + [ input_vars['plot'][axis] for axis in ['x','y'] ]
        super(PrepareInputsDialog, self).__init__( None, DialogBase.DATA_PREP, partial( prepare_inputs, input_vars, subsample ), scope, **kwargs )

    def addFileContent( self, inputsLayout: QBoxLayout ):
        for input_file_id in self.inputs:
            inputsLayout.addLayout( self.createFileSystemSelectionWidget( input_file_id, self.FILE, f"data/init/{input_file_id}", "data/dir" ) )

    def getProjectList(self) -> Optional[List[str]]:
        system_settings = dataManager.getSettings( QSettings.SystemScope )
        settings_file = system_settings.fileName()
        settings_path = os.path.dirname( os.path.realpath( settings_file ) )
        inifiles = glob.glob(f"{settings_path}/*.ini")
        sorted_inifiles = sorted( inifiles, key=lambda t: os.stat(t).st_mtime )
        return [ os.path.splitext( os.path.basename( f ) )[0] for f in sorted_inifiles ]



