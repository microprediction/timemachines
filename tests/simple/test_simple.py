from timemachines.skaters.simple.movingaverage import precision_ema_ensemble, aggressive_ema_ensemble
from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit, hospital_exog_mean_square_error_with_sporadic_fit

SIMPLE_TO_TEST = [ precision_ema_ensemble, aggressive_ema_ensemble ]


from timemachines.inclusion.skleaninclusion import using_sklean
if using_sklean:
    def test_ensemble_errors():
        for f in SIMPLE_TO_TEST:
            err = hospital_mean_square_error_with_sporadic_fit(f=f, k=5, n=150, fit_frequency=1)



if __name__=='__main__':
    assert using_sklean
    test_ensemble_errors()