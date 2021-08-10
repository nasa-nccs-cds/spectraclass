import numpy as np
import os
from typing import List, Union, Tuple, Optional, Dict, Callable, Iterable
from torch.utils.data import DataLoader
import torch
import xarray as xa
from torch.utils.data import Dataset
from torch import Tensor

class xaTorchDataset( Dataset ):
    def __init__(self, data: xa.DataArray, labels: xa.DataArray = None, transform: Callable = None ):
        super(xaTorchDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._transform = transform

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx) -> Tuple[Tensor,Tensor]:
        data = torch.from_numpy( self._data[idx].values.astype( "float64" ) )
        label = None if self._labels is None else torch.from_numpy( self._labels[idx].values.astype( "float64" ) )
        if self._transform:
            data = self._transform(data)
        return data, label


