table_type = "bokeh"
from spectraclass.util.logs import LogManager, lgm

def tbm():
    lgm().log( f"Creating {table_type} table")
    if table_type == "bokeh":
        from .bktable import TableManager
        return TableManager.instance()
    if table_type == "ipsheet":
        from .iptable import TableManager
        return TableManager.instance()
    if table_type == "qgrid":
        from .qtable import TableManager
        return TableManager.instance()