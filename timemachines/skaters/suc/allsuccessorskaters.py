from timemachines.skaters.suc.successorinclusion import using_successor

if using_successor:
    from successor.skaters.scalarskaters.allscalarskaters import SCALAR_SKATERS
    SUCCESSOR_SKATERS = SCALAR_SKATERS
else:
    SUCCESSOR_SKATERS = []