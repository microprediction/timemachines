from timemachines.skaters.rvr.rvrinclusion import using_river
from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE, wrap
if using_river:
    from river import linear_model, optim, preprocessing, time_series
    from timemachines.skatertools.components.parade import parade
    from timemachines.skatertools.utilities.nonemath import nonecast, nonecenter

    def rvr_sarimax_factory(y :Y_TYPE, s, k:int, a:A_TYPE =None,
                            t:T_TYPE =None, e:E_TYPE =None, r:R_TYPE=None,
                            p:int=0, d:int=0, q:int=0, m=1, sp:int=0, sq:int=0,
                            intercept_init=110, optim_sgd=0.01, intercept_lr=0.3 ):
        y = wrap(y)
        a = wrap(a)

        if not s.get('k'):
            s['k']=k
            s['n'] = len(y)
            s['p']={}  # parade
            s['model']=None
        else:
            assert len(y) == s['n']
            assert k == s['k']

        if y is None:
            return None, s, None
        else:
            model = s.get('model')
            if model is None:
                model = time_series.SNARIMAX(p=p, d=d, q=q, m=m, sp=sp, sq=sq,
                                              regressor=(
                                                      preprocessing.StandardScaler() |
                                                      linear_model.LinearRegression(
                                                          intercept_init=intercept_init,
                                                          optimizer=optim.SGD(optim_sgd),
                                                          intercept_lr=intercept_lr
                                                      )
                                              ))
            x = model.forecast(horizon=1)
            _we_ignore_bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y[0])
            x_std_fallback = nonecast(x_std, fill_value=1.0)
            model = model.learn_one(x=None,y=y[0])
            s['model'] = model
            return x, x_std_fallback, s


