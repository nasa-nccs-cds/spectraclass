table_type = "bokeh"

def tbm():
    if table_type == "bokeh":
        from .bktable import TableManager
        return TableManager.instance()
    if table_type == "ipsheet":
        from .iptable import TableManager
        return TableManager.instance()
    if table_type == "qgrid":
        from .qtable import TableManager
        return TableManager.instance()