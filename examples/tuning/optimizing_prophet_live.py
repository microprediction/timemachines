# Illustrates how to find the best hyper-parameter r in (0,1), and interpret this as two prophet hyper-parameters
# We use a random stream from https://www.microprediction.org/browse_streams.html
# Your should expect this to take a while. A time update is provided after the first function evaluation.


if __name__=='__main__':
    # We need quit a few optional install for this example
    print("Prophet will be fit for most of them, after a burn_in, and for many different hyper-params. Don't hold your breathe.")
    from timemachines.inclusion.humpdayinclusion import using_humpday
    from timemachines.inclusion.micropredictioninclusion import using_microprediction
    from timemachines.skaters.proph.prophetinclusion import using_prophet
    assert using_prophet,'pip install prophet'
    assert using_humpday,'pip install humpday'
    assert using_microprediction,'pip install microprediction'
    try:
        from humpday.optimizers.optunacube import optuna_tpe_cube
    except ImportError:
        raise ValueError('pip install optuna')
    from timemachines.skatertools.tuning.hyperempirical import optimal_r_for_stream
    from timemachines.skaters.proph.prophskaterssingular import fbprophet_univariate_r2
    from timemachines.skaters.proph.prophparams import PROPHET_META, prophet_params
    from pprint import pprint
    from timemachines.skatertools.data.live import random_regular_stream_name


    # ... but then the rest is easy
    name, url = random_regular_stream_name(min_len=PROPHET_META['n_warm'], with_url=True)
    print('We will find the best fbprophet hyper-parameters for ' + url)
    best_r, best_value, info = optimal_r_for_stream(f=fbprophet_univariate_r2,name=name,k=10,optimizer=optuna_tpe_cube,
                                                    n_burn=PROPHET_META['n_warm']+20,n_trials=50,n_dim=2)
    pprint(info)
    params = prophet_params(r=best_r,dim=2)
    pprint(params)


