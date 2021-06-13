from timemachines.skaters.divine.divineinclusion import using_divinity, dv
if using_divinity:
    import numpy as np
    from timemachines.skatertools.utilities.conventions import Y_TYPE, K_TYPE
    from timemachines.skatertools.utilities.suppression import no_stdout_stderr
    from timemachines.skatertools.visualization.priorplot import prior_plot
    from timemachines.skatertools.utilities.conventions import wrap
    from timemachines.skatertools.components.parade import parade
    from timemachines.skatertools.utilities.nonemath import nonecast
    from timemachines.skaters.divine.divineparams import MIN_N_WARM, DIVINE_MODEL
    from copy import deepcopy

    # This guy is a little chatty

    def divinity_univariate_factory(y:Y_TYPE, s, k:K_TYPE, a=None, t=None, e=None,
                                    max_buffer_len=1000,
                                    n_warm = 101,
                                    model_params:dict=None):
        """ A partial wrapping of the divinity library with notable limitations:

             - Fits every invocation
             - Ignores exogenous variables
             - State is merely a buffer

        """
        y0 = wrap(y)[0]
        assert n_warm>=101,' You must use n_warm'

        if not s:
            s = dict(y=[])

        if not s.get('p'):
            s['p'] = {}   # Initialize prediction parade

        if y0 is None:
            return None, None, s   # Ignore suggestion to fit offline

        # Update buffer
        s['y'].append(y0)
        if len(s['y'])>max_buffer_len+1000:
            s['y'] = s['y'][-max_buffer_len:]

        # Fit and predict, if warm, or just last value
        if len(s['y']) < max( n_warm, MIN_N_WARM ):
            return [y0]*k, [abs(y0)]*k, s
        else:
            with no_stdout_stderr():
                kwargs = deepcopy(DIVINE_MODEL)
                if model_params:
                    kwargs.update(**model_params)
                model = dv.divinity(forecast_length=k,**kwargs)
                model.fit(np.array(s['y']))
            x = list(model.predict())
            _we_ignore_bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0)
            x_std_fallback = nonecast(x_std, fill_value=1.0)
            return x, x_std_fallback, s


if __name__=='__main__':
    assert using_divinity,'pip install divinity'
    if using_divinity:
        err = prior_plot(f=divinity_univariate_factory, k=5, n=500)
        pass
    else:
        print('pip install divinity')

