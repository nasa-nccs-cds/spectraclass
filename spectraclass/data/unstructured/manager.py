from spectraclass.data.base import ModeDataManager
from .modes import *
import numpy as np
import os, pickle


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

    def execute_task( self, task: str ):
        from spectraclass.gui.points import PointCloudManager
        from spectraclass.gui.unstructured.table import TableManager
        from spectraclass.reduction.embedding import ReductionManager
        tmgr = TableManager.instance()
        if task == "embed":
            embedding = ReductionManager.instance().umap_embedding()
            PointCloudManager.instance().reembed( points = embedding )
        elif task == "mark":
            tmgr.mark_selection()
        elif task == "spread":
            tmgr.spread_selection()
        elif task == "clear":
            tmgr.clear_current_class()
        elif task == "undo":
            tmgr.undo_action()
        elif task == "distance":
            tmgr.display_distance()

