from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, R_TYPE, E_TYPE, T_TYPE
from timemachines.skatertools.components.parade import parade
from timemachines.skatertools.utilities.nonemath import nonecast, nonecenter
from timemachines.skaters.bats.batsinclusion import using_bats
from timemachines.skatertools.utilities.conventions import wrap
from timemachines.skaters.bats.batsifactory import bats_iskater_factory
import numpy as np
from timemachines.skatertools.batch.batchskater import batch_skater_factory


if using_bats:
    from tbats import TBATS

    def bats_factory(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r:R_TYPE=None,
                     emp_mass: float = 0.0,
                     use_box_cox=None, box_cox_bounds=(0, 1),
                     use_trend=None, use_damped_trend=None,
                     seasonal_periods=None, use_arma_errors=True
                     ):
        return batch_skater_factory(y=y,s=s,k=k,a=a,t=t,e=e,r=r,
                                    iskater=bats_iskater_factory,
                                    iskater_kwargs={'use_box_cox':use_box_cox,
                                                    'box_cox_bounds':box_cox_bounds,
                                                    'use_trend':use_trend,
                                                    'use_damped_trend':use_damped_trend,
                                                     'seasonal_periods':seasonal_periods,
                                                    'use_arma_errors':use_arma_errors},
                                    min_e=0, emp_mass=emp_mass, emp_std_mass=1.0, n_warm=10)


    def bats_factory_old(y :Y_TYPE, s, k:int, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None, r:R_TYPE=None,
                     emp_mass: float = 0.0,
                     use_box_cox=None, box_cox_bounds=(0, 1),
                     use_trend=None, use_damped_trend=None,
                     seasonal_periods=None, use_arma_errors=True
                     ):
        """
          :param emp_mass:        How much to use empirical bias to correct
        """
        assert 0<= emp_mass<=1

        if e is None:
           e = 1  # 'Providing e=None to bats is not recommended. Use e<0 to skip a fitting or e>0 to fit.'

        y = wrap(y)
        a = wrap(a)

        if not s.get('y'):
            s = {'p': {},      # parade
                 'y': list(),  # historical y
                 'a': list(),  # list of a known k steps in advance
                 't': list(),
                 'k': k}
        else:
            # Assert immutability of k, dimensions of y,a
            if s['y']:
                assert len(y) == len(s['y'][0])
                assert k == s['k']
            if s['a']:
                assert len(a) == len(s['a'][0])

        if y is None:
            return None, s, None
        else:
            s['y'].append(y)
            if a is not None:
                s['a'].append(a)
            if t is not None:
                assert isinstance(t,float), 'epoch time please'
                s['t'].append(t)

            if (e is not None) and (e>0) and ((a is None) or (len(s['a'])>k)) and (len(s['y'])>10):
                # Offset y, t, a are supplied to prophet interface
                # Prophet requires several non-nan rows
                t_arg = s['t'][k:] if t is not None else None
                a_arg = s['a']
                y_arg = s['y'][k:]
                x, x_std = bats_iskater_factory(y=y_arg, k=k, a=a_arg, t=t_arg,
                                                                 use_box_cox=use_box_cox, box_cox_bounds=box_cox_bounds,
                                                                 use_trend=use_trend, use_damped_trend=use_damped_trend,
                                                                 seasonal_periods=seasonal_periods, use_arma_errors=use_arma_errors
                                                                 )
                s['m'] = True # Flag indicating a model has been fit (there is no point keeping the model itself, however)
            else:
                x = [y[0]] * k
                x_std = [1] * k

            # Get running mean prediction errors from the prediction parade
            x_resid, x_resid_std, s['p'] = parade(p=s['p'], x=x, y=y[0])
            x_resid_std = nonecast(x_resid_std,1.0)

            # Compute center of mass between bias-corrected and uncorrected predictions
            emp_std_mass = 1.0 # <--- Makes no sense to use x_std_rubbish as the model doesn't compute it
            x_corrected = np.array(x_resid) + np.array(x)
            x_center = nonecenter(m=[emp_mass, 1 - emp_mass], x=[x_corrected, x])
            x_std_center = nonecenter(m=[emp_std_mass, 1 - emp_std_mass], x=[x_resid_std, x_std])

            return x_center, x_std_center, s