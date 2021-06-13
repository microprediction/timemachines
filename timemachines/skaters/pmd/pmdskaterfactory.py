from timemachines.skaters.pmd.pmdinclusion import using_pmd, pm
if using_pmd:

    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE, S_TYPE
    from timemachines.skatertools.components.observance import observance
    from timemachines.skatertools.utilities.conventions import split_exogenous, wrap, dimension
    from timemachines.skaters.pmd.pmddefaultparams import pmd_params
    import numpy as np
    from typing import Any
    from timemachines.skatertools.components.chronometer import tick, tock, tocks
    from pprint import pprint
    from typing import Union, List


    ###################################################################################################
    #                                                                                                 #
    #                      PMD-ARIMA skater factory                                                   #
    #                                                                                                 #
    # As with all skaters, the intent is that you can cycle through observations as follows:          #
    #                                                                                                 #
    #      for yi, ai in zip(y, a):                                                                   #
    #         x, x_std, s = pmd_exogenous(y=yi, s=s, k=7, a=ai)                                       #
    #                                                                                                 #
    # where each x and s_std are length 7, in this example, and a[0] is contemporaneous with y[7]     #
    #                                                                                                 #
    # Advantages:                                                                                     #
    #       - Pretty good amortized invocation time, say if you fit every 100 invocations             #
    # Disadvantages:                                                                                  #
    #       - Model state is opaque or at best obscure                                                #
    #                                                                                                 #
    ###################################################################################################


    def pmd_skater_factory(y:Y_TYPE, s:dict, k:int=1, a:A_TYPE=None, t:T_TYPE=None, e:E_TYPE=None,
                           method: str= 'default', n_warm=50,
                           model_params:dict=None)->(Union[List[float],None],
                                                     Union[List[float],None], Any):
        """ Predict using both simultaneously observed and known in advance variables
            y: Y_TYPE    scalar or list where y[1:] are interpreted as contemporaneously observed exogenous variables
            s:           state
            k:           Number of steps ahead to predict
            a:           (optional) scalar or list of variables known k-steps in advance.
                         When calling, provide the known variable k steps ahead, not the contemporaneous one.
            t:           (optional) Time of observation.
            e:           (optional) Maximum computation time (supply e>60 to give hint to do fitting)

            :returns: x [float] , s', scale [float]

            Remarks:
               - Model params cannot be changed after the first invocation.
               - Allows y=None to be used
        """
        y = wrap(y)
        a = wrap(a)

        if not s.get('n_obs'):
            # Initialize
            s['n_obs'] = 0
            s['model'] = None
            s['immutable'] = pmd_set_immutable(k=k, y=y, a=a, n_warm=n_warm)
            s['params'] = pmd_params(method=method)
            if model_params:
                s['params'].update(model_params)
            s['o'] = dict()                         # Observance
        else:
            pmd_check_consistent_usage(y=y,s=s,a=a,k=k)

        tick(s)
        if t is not None:
            pass # Other models might perform an evolution step here. Not applicable to PMDARIMA

        if y is not None:
            # Receive observation y[0], possibly exogenous y[1:] and possibly k-in-advance a[:]
            # Collect from queues the contemporaneous variables
            s['n_obs']+=1
            y_t, z = split_exogenous(y)
            x_t, s['o'] = observance(y=y,o=s['o'],k=k,a=a)

            # Update the pmdarima model itself
            if x_t is not None:
                if s['model'] is not None:
                    if x_t:
                        s['model'].update([y_t], [x_t])
                    else:
                        s['model'].update([y_t])

            # Predict
            if s['model'] is None:
                # Fall back to last value if there is no model calibrated as yet
                x = [y_t]*k
                if len(s['o']['x']) > 5 + 2*k:
                    Y = s['o']['y'][k+1:]
                    X = s['o']['x'][k+1:]
                    x_std = [ np.nanstd( [ xi[0]-yk[0] for xi, yk in zip( X, Y[j:] ) ] ) for j in range(1,k+1) ]
                else:
                    x_std = [1.0]*k   # Fallback to dreadful estimate
            else:
                # Predict forward, supplying known data if it exists
                if not a and not z:
                    z_forward = None
                else:
                    if not a:
                        z_forward = [z]*k
                    else:
                        z_forward = [ list(z) + list(ai) for ai in s['o']['a'] ]  # Add known k-steps ahead
                                        # This estimate could be improved by predicting z's and attenuating
                                        # It is only really a good idea for k=1
                x, ntvls = s['model'].predict(n_periods=k, X=z_forward, return_conf_int=True, alpha=s['immutable']['alpha'])
                x_std = list([ ntvl[1] - ntvl[0] for ntvl in ntvls ])

        # Fit
        tock(s)
        if pmd_it_is_time_to_fit(s=s, e=e):
            tick(s)
            X = s['o'].get('x') or None
            Y = s['o']['y']
            s['model'] = pm.auto_arima(y=Y, X=X, **s['params'])
            print(s['model'])
            print(s['model'].arima_res_.params)
            pprint(tocks(s))
            tock(s,'fit')
            pprint(tocks(s))

        if y is not None:
            return list(x), list(x_std), s
        else:
            return None, None, s


    def pmd_it_is_time_to_fit(s:S_TYPE, e:E_TYPE)->bool:
        """ Provided 60 seconds, or getting stale """
        return s['n_obs'] == s['immutable']['n_warm'] or \
               (s['n_obs'] > s['immutable']['n_warm'] and ((e is not None and e>60) or s['n_obs'] % s['immutable']['n_fit'] == 0))


    def pmd_set_immutable(y: Y_TYPE, k:int, a:A_TYPE=None, n_warm:int=20):
        """ Set on the first invocation, when s={} is passed """
        return {'k': k,
                'alpha':0.25,   # Determines confidence interval
                'n_fit':250,
                'n_warm':n_warm,
                'dim_exog': dimension(y) - 1,
                'dim_a': dimension(a)}


    def pmd_check_consistent_usage(y:Y_TYPE,s,k,a):
        if y is not None:
            assert dimension(y)-1 == s['immutable']['dim_exog']
        if k is not None:
            assert k==s['immutable']['k']
        if a is not None:
            assert dimension(a)==s['immutable']['dim_a']




