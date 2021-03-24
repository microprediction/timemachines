from timemachines.skatertools.tuning import optimal_r_for_stream
from timemachines.skaters.nproph.nprophskaterssingular import fbnprophet_univariate_r2
from humpday.optimizers.dlibcube import dlib_default_cube
from microprediction import MicroReader
from timemachines.skaters.nproph.nprophparams import NPROPHET_META, nprophet_params
import random
from pprint import pprint


# Optimal meta-parameters for nprophet skaters


def fbnprophet_univariate_best_r()->float:
    """
      :return:  Provides best r for a randomly chosen data stream
                Takes about 12 hrs to run
    """
    mr = MicroReader()
    names = mr.get_stream_names()
    okay = False
    while not okay:
        name = random.choice(names)
        n_obs = len(mr.get_lagged_values(name=name))
        okay = n_obs > NPROPHET_META['n_warm']+50 and '~' not in name
    url = 'https://www.microprediction.org/stream_dashboard.html?stream='+name.replace('.json','')
    print('We will find the best fbnprophet hyper-parameters for '+url)
    print('There are ' + str(n_obs) + ' observations in the series.')
    print("Prophet will be fit for most of them, after a burn_in, and for many different hyper-params. Don't hold your breathe.")

    best_r, best_value, info = optimal_r_for_stream(f=fbnprophet_univariate_r2,name=name,k=10,optimizer=dlib_default_cube,
                                                    n_burn=NPROPHET_META['n_warm']+20,n_trials=50,n_dim=2)
    pprint(info)
    params = nprophet_params(r=best_r,dim=2)
    pprint(params)