from spectraclass.data.base import DataManager
import sys

if len(sys.argv) != 3:
    print( f"Usage: {sys.argv[0]} <project> <mode>")
else:

    project: str = sys.argv[1]    #   e.g. 'demo1', 'demo2', 'demo3', or 'demo4'
    mode: str = sys.argv[2]       #   e.g. 'swift', 'tess', 'desis', or 'aviris'

    dm: DataManager = DataManager.initialize( project, mode )
    dm.prepare_inputs()
    dm.save_config()

