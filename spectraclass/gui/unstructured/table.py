table_type = "qgrid"

def tm():
    if table_type == "bokeh":
        from .bktable import TableManager
        return TableManager.instance()
    if table_type == "qgrid":
        from .qtable import TableManager
        return TableManager.instance()