table_type = "ipysheet"
from spectraclass.util.logs import LogManager, lgm

def tbm():
    lgm().log( f"Creating {table_type} table")
    if table_type == "bokeh":
        from .bktable import TableManager
        return TableManager.instance()
    if table_type == "ipysheet":
        from .iptable import TableManager
        return TableManager.instance()
    if table_type == "qgrid":
        from .qtable import TableManager
        return TableManager.instance()