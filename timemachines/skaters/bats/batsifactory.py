from typing import List, Tuple, Any
from timemachines.skatertools.utilities.conventions import wrap
from timemachines.skatertools.utilities.suppression import no_stdout_stderr
from timemachines.skaters.bats.batsinclusion import using_bats

if using_bats:
    from tbats import TBATS

    def bats_iskater_factory(y: [[float]], k: int, a: List = None, t: List = None, e=None, freq: str = None, n_max=1000,
                             use_box_cox=None, box_cox_bounds=(0, 1),
                             use_trend=None, use_damped_trend=None,
                             seasonal_periods=None, use_arma_errors=True):
        if a:
            assert len(a) == len(y) + k

        if isinstance(y[0], float):
            y = [wrap(yj) for yj in y]

        estimator = TBATS(use_box_cox=use_box_cox, box_cox_bounds=box_cox_bounds,
                          use_trend=use_trend, use_damped_trend=use_damped_trend,
                          seasonal_periods=seasonal_periods, use_arma_errors=use_arma_errors)
        with no_stdout_stderr():
            fitted_model = estimator.fit(y)
        x = list(fitted_model.forecast(steps=k))
        x_std = [1.0 for _ in x]
        return x, x_std


if __name__=='__main__':
    assert using_bats, 'pip install tbats'
    import numpy as np
    import time
    st = time.time()
    y = list(np.cumsum(np.random.randn(400)))
    x, x_std = bats_iskater_factory(y=y,k=5)
    print(x)
    print(time.time()-st)
