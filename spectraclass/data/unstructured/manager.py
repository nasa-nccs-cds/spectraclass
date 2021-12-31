from spectraclass.data.base import ModeDataManager
from collections import OrderedDict
from spectraclass.reduction.embedding import ReductionManager, rm
from pathlib import Path
from spectraclass.util.logs import LogManager, lgm, exception_handled, log_timing
import xarray as xa
from typing import List, Union, Tuple, Optional, Dict, Callable
import traitlets as tl
import numpy as np
import os, pickle

class UnstructuredDataManager(ModeDataManager):

    def __init__(self):
        super(UnstructuredDataManager, self).__init__()
        self._dsid = self.INPUTS['embedding']
        self._cached_data = {}

    @exception_handled
    def prepare_inputs(self) -> Dict[Tuple,int]:
        self.update_gui_parameters()
        self.set_progress(0.02)
        output_file = os.path.join(self.datasetDir, self.dsid() + ".nc")
        assert (self.INPUTS is not None), f"INPUTS undefined for mode {self.mode}"

        np_embedding: np.ndarray = self.getInputFileData( )
        if np_embedding is not None:
            dims = np_embedding.shape
            mdata_vars = list(self.INPUTS['directory'])
            xcoords = OrderedDict(samples=np.arange(dims[0]), bands=np.arange(dims[1]))
            xdims = OrderedDict({dims[0]: 'samples', dims[1]: 'bands'})
            data_vars = dict( embedding=xa.DataArray(np_embedding, dims=xcoords.keys(), coords=xcoords, name=self.INPUTS['embedding']) )
            data_vars.update({vid: self.getXarray(vid, xcoords, self.subsample, xdims) for vid in mdata_vars})
            pspec = self.INPUTS['plot']
            data_vars.update(  {f'plot-{vid}': self.getXarray(pspec[vid], xcoords, self.subsample, xdims, norm=pspec.get('norm','spectral')) for vid in ['x', 'y'] } )
            self.set_progress(0.1)
            if self.reduce_method and (self.reduce_method.lower() != "none"):
                input_data = data_vars['embedding']
                ( reduced_spectra, reproduced_spectra, usable_input_data ) = rm().reduce(input_data, None, self.reduce_method, self.model_dims, self.reduce_nepochs, self.reduce_sparsity)[0]
                coords = dict(samples=xcoords['samples'], model=np.arange(self.model_dims))
                data_vars['reduction'] = xa.DataArray(reduced_spectra, dims=['samples', 'model'], coords=coords)
                data_vars['reproduction'] = usable_input_data.copy(data=reproduced_spectra)
                self.set_progress(0.8)

            result_dataset = xa.Dataset(data_vars, coords=xcoords, attrs={'type': 'spectra'})
            result_dataset.attrs["colnames"] = mdata_vars
            if os.path.exists(output_file): os.remove(output_file)
            lgm().log( f"Writing output to {output_file}", print=True )
            result_dataset.to_netcdf(output_file, format='NETCDF4', engine='netcdf4')
            self.updateDatasetList()
            self.set_progress(1.0)
            return {}
        else:
            print( "DATA PRE-PROCESSING FAILED!  See log file for more info.")


    @exception_handled
    def getInputFileData( self, vname: str = None, **kwargs ) -> np.ndarray:
        if vname is None: vname = self.INPUTS['embedding']
        if 'subsample' in kwargs: self.subsample = kwargs.get('subsample')
        input_file_path = os.path.expanduser( os.path.join(self.data_dir, self.mode, f"{vname}.pkl"))
        if os.path.isfile(input_file_path):
            input_data = self._cached_data.get( input_file_path, None )
            if input_data is None:
                with open(input_file_path, 'rb') as f:
                    result = pickle.load(f)
                    if isinstance(result, np.ndarray):
                        skip_subsample =  (( result.shape[0] == self.model_dims ) and result.ndim == 1) or (self.subsample == 1)
                        input_data =  result if skip_subsample else result[::self.subsample ]
                    elif isinstance(result, list):
                        subsampled = [result[i] for i in range(0, len(result), self.subsample )]
                        input_data = np.vstack(subsampled) if isinstance(result[0], np.ndarray) else np.array(subsampled)
                lgm().log(  f"Reading unstructured {vname} data from file {input_file_path}, shape = {input_data.shape}", print=True )
                self._cached_data[input_file_path] = input_data
            return input_data
        else:
            lgm().log( f"Error, the input path '{input_file_path}' is not a file.", print=True )

    def dsid(self, **kwargs) -> str:
        return self._dsid

