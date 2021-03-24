from timemachines.skatertools.tuning import optimal_r_for_stream
from timemachines.skaters.proph.prophskaterssingular import fbprophet_univariate_r2
from humpday.optimizers.dlibcube import dlib_default_cube
from microprediction import MicroReader
from timemachines.skaters.proph.prophparams import PROPHET_META, prophet_params
import random
from pprint import pprint


# Optimal meta-parameters for prophet skaters


def fbprophet_univariate_best_r()->float:
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
        okay = n_obs > PROPHET_META['n_warm']+50 and '~' not in name
    url = 'https://www.microprediction.org/stream_dashboard.html?stream='+name.replace('.json','')
    print('We will find the best fbprophet hyper-parameters for '+url)
    print('There are ' + str(n_obs) + ' observations in the series.')
    print("Prophet will be fit for most of them, after a burn_in, and for many different hyper-params. Don't hold your breathe.")

    best_r, best_value, info = optimal_r_for_stream(f=fbprophet_univariate_r2,name=name,k=10,optimizer=dlib_default_cube,
                                                    n_burn=PROPHET_META['n_warm']+20,n_trials=50,n_dim=2)
    pprint(info)
    params = prophet_params(r=best_r,dim=2)
    pprint(params)