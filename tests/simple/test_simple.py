from timemachines.skaters.simple.movingaverage import precision_ema_ensemble, aggressive_ema_ensemble

SIMPLE_TO_TEST = [ precision_ema_ensemble, aggressive_ema_ensemble ]


from timemachines.inclusion.sklearninclusion import using_sklearn
if using_sklearn:
    from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit, \
        hospital_exog_mean_square_error_with_sporadic_fit

    def test_ensemble_errors():
        for f in SIMPLE_TO_TEST:
            err = hospital_mean_square_error_with_sporadic_fit(f=f, k=5, n=150, fit_frequency=1)





if __name__=='__main__':
    assert using_sklearn
    test_ensemble_errors()