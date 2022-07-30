from spectraclass.model.base import SCSingletonConfigurable
from typing import List, Union, Tuple, Optional, Dict, Type
import ipywidgets as ipw

class CustomVisualization:

    def gui(self) -> ipw.DOMWidget:
        pass

class CustomVizManager(SCSingletonConfigurable):

    def __init__(self):
        SCSingletonConfigurable.__init__(self)
        self._visuals: Dict[str,CustomVisualization] = {}

    def addVisualization(self, vname: str, viz: CustomVisualization):
        self._visuals[vname] = viz

    @property
    def names(self) -> List[str]:
        return list(self._visuals.keys())

    @property
    def guis(self) -> List[CustomVisualization]:
        return list(self._visuals.values())

    def __getitem__(self, vname ) -> CustomVisualization:
        return self._visuals.get(vname)


def cviz():
    return CustomVizManager.instance()