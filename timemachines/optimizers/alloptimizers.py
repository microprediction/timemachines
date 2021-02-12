#!/usr/bin/env python

# add command line args, clean up output a little, add timings, collect results into table
from timemachines.optimizers.hyperoptcube import HYPEROPT_OPTIMIZERS
from timemachines.optimizers.shgocube import SHGO_OPTIMIZERS
from timemachines.optimizers.optunacube import OPTUNA_OPTIMIZERS
from timemachines.optimizers.pysotcube import PYSOT_OPTIMIZERS
from timemachines.optimizers.scipycube import SCIPY_OPTIMIZERS
from timemachines.optimizers.axcube import AX_OPTIMIZERS
from timemachines.optimizers.platypuscube import PLATYPUS_OPTIMIZERS
from timemachines.optimizers.pymoocube import PYMOO_OPTMIZERS
from timemachines.optimizers.swarmlibcube import SWARMLIB_OPTIZERS
from timemachines.optimizers.nevergradcube import NEVERGRAD_OPTIMIZERS
from timemachines.objectives.classic import CLASSIC_OBJECTIVES

from datetime import datetime
import pandas as pd
import argparse

CANDIDATES = SCIPY_OPTIMIZERS + SHGO_OPTIMIZERS + HYPEROPT_OPTIMIZERS +\
             PYSOT_OPTIMIZERS + OPTUNA_OPTIMIZERS + AX_OPTIMIZERS +\
             PLATYPUS_OPTIMIZERS + PYMOO_OPTMIZERS + NEVERGRAD_OPTIMIZERS + SWARMLIB_OPTIZERS

# To see what might be working, or not, refer regression testing results in directories such as:
# https://github.com/microprediction/timemachines-testing/tree/main/data/brownian/dlm_seasonal
OPTIMIZERS = SHGO_OPTIMIZERS + SCIPY_OPTIMIZERS + PYSOT_OPTIMIZERS + AX_OPTIMIZERS + \
             OPTUNA_OPTIMIZERS + PLATYPUS_OPTIMIZERS + NEVERGRAD_OPTIMIZERS +\
             SWARMLIB_OPTIZERS + HYPEROPT_OPTIMIZERS + PYMOO_OPTMIZERS


def optimizer_from_name(name):
    valid = [f for f in OPTIMIZERS if f.__name__==name ]
    return valid[0] if len(valid)==1 else None


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Run all optimizers on input size ndim (default 2) requesting ntrials (default 20) iterations and save results (default log.csv")

    parser.add_argument("-d", "--ndims", type=int, action="extend", nargs="+",
                        help="Number of input dimensions to objective function (default 2)")
    parser.add_argument("-t", "--ntrials", type=int, action="extend", nargs="+",
                        help="Number of trial iterations in optimization (default 20)")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                        action="store_true")
    parser.add_argument("-o", "--logfile", help="Specify outputfile (default log.csv)")
    args = parser.parse_args()
    # print(args)
    LOGFILE = 'log.csv'
    if args.logfile is not None:
        LOGFILE = args.logfile

    NDIMS = [2]
    if args.ndims is not None:
        NDIMS = args.ndims
        
    NTRIALS = [20]
    if args.ntrials is not None:
        NTRIALS = args.ntrials

    print(' ')
    print('Full list of optimizer strategies .. ')
    print(' ')
    print([ o.__name__.replace('_cube','') for o in OPTIMIZERS])
    print(' ')
    print('Full list of objective functions .. ')
    print(' ')
    print([o.__name__ for o in CLASSIC_OBJECTIVES])
    print(' ')

    print(datetime.now(), str(len(OPTIMIZERS)) + ' optimization strategies will be compared.')
    print(datetime.now(), str(len(CLASSIC_OBJECTIVES)) + ' objective functions will be employed.')
    print(datetime.now(), 'objective input dimensions: ', str(NDIMS))
    print(datetime.now(), 'number of trials: ', str(NTRIALS))
    print(datetime.now(), 'logfile: ', LOGFILE)

    log_array = []
    for objective in CLASSIC_OBJECTIVES:
        print(' ')
        for n_dim in NDIMS:
            for n_trials in NTRIALS:
                print(datetime.now(), 'Now testing against '+objective.__name__+' in '+str(n_dim)+' dimensions requesting '+str(n_trials)+' trials.')
                for optimizer in OPTIMIZERS:
                    try:
                        start_time = datetime.now()
                        result = optimizer(objective, n_trials=n_trials, n_dim=n_dim, with_count=True)
                        best_value, best_params, reported_trials = result
                        end_time = datetime.now()
                        time_elapsed = datetime.now() - start_time
                        print(datetime.now(), "Finished", reported_trials, "trials in ", time_elapsed)
                        print(datetime.now(), optimizer.__name__, result)
                        log_array.append([start_time, end_time, time_elapsed,
                                          objective.__name__, optimizer.__name__,
                                          n_dim, n_trials, reported_trials,
                                          best_value, best_params, ])
                    except:
                        import warnings
                        print(' ')
                        warnings.warn(' WARNING : '+optimizer.__name__+' fails on '+ objective.__name__+ ' in '+str(n_dim)+' dimensions with '+str(n_trials)+' trials.')

    log_df = pd.DataFrame(log_array)
    log_df.columns = ["start_time", "end_time", "time_elapsed",
                      "objective", "optimizer",
                      "n_dim", "n_trials", "reported_trials",
                      "best_value", "best_params", ]
    print(log_df)
    log_df.to_csv(LOGFILE, index=False)
