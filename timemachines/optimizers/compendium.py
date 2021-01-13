from timemachines.optimizers.hyperoptcube import hyperopt_cube
from timemachines.optimizers.shgocube import shgo_cube
from timemachines.optimizers.optunacube import optuna_cube
from timemachines.optimizers.pysotcube import pysot_cube
from timemachines.optimizers.powellcube import powell_cube
from timemachines.optimizers.axcube import ax_cube
from timemachines.optimizers.platypuscube import genetic_cube, evolutionary_cube
from timemachines.optimizers.pymoocube import nelder_cube, ctaea_cube, nsga3_cube, pattern_cube

OPTIMIZERS = [ ax_cube, hyperopt_cube, shgo_cube, optuna_cube, pysot_cube,
               powell_cube, evolutionary_cube, genetic_cube, nelder_cube,
                nsga3_cube, pattern_cube ]

FAILING_OPTIMIZERS = [ ctaea_cube ]

if __name__=='__main__':
    from timemachines.optimizers.objectives import AN_OBJECTIVE
    minima = [ optimizer(AN_OBJECTIVE, n_trials=50, n_dim=5) for optimizer in OPTIMIZERS]