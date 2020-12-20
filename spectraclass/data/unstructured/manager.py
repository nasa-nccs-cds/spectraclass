from spectraclass.data.base import DataManager, ModeDataManager
from .modes import *
import numpy as np
from typing import List, Union, Tuple, Optional, Dict
import os, math, pickle, glob

class UnstructuredDataManager(ModeDataManager):

    def __init__(self):
        super(UnstructuredDataManager, self).__init__()


    def getInputFileData( self ) -> np.ndarray:
        input_file_path = os.path.expanduser(
            os.path.join(self.data_dir, self.dm.name, self.config_mode, f"{self.dataset}.pkl"))
        try:
            if os.path.isfile(input_file_path):
                print(f"Reading unstructured {self.dataset} data from file {input_file_path}, dims = {self.model_dims}")
                with open(input_file_path, 'rb') as f:
                    result = pickle.load(f)
                    if isinstance(result, np.ndarray):
                        if  (result.shape[0] == self.model_dims[1]) and result.ndim == 1: return result
                        return result[::self.subsample ]
                    elif isinstance(result, list):
                        #                        if dims is not None and ( len(result) == dims[1] ): return result
                        subsampled = [result[i] for i in range(0, len(result), self.subsample )]
                        if isinstance(result[0], np.ndarray):
                            return np.vstack(subsampled)
                        else:
                            return np.array(subsampled)
            else:
                print(f"Error, the input path '{input_file_path}' is not a file.")
        except Exception as err:
            print(f" Can't read data[{self.dataset}] file {input_file_path}: {err}")