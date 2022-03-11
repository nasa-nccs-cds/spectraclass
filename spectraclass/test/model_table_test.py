from spectraclass.learn.manager import ModelTable
from spectraclass.util.logs import lgm
lgm().init_logging( 'model_table_test', 'debug' )

models = dict( test1 = "/tmp", test2 = "/tmp", test3 = "/tmp" )
table = ModelTable( models )

table.gui()