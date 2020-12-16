from spectraclass.data.base import DataManager, ModeDataManager
from .modes import *

class UnstructuredDataManager(ModeDataManager):

    def __init__(self):
        super(UnstructuredDataManager, self).__init__()