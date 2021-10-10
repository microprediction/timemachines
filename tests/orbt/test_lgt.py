from timemachines.skaters.orbt.allorbitskaters import ORBIT_LGT_SKATERS
from timemachines.skatertools.evaluation.evaluators import hospital_mean_square_error_with_sporadic_fit
from timemachines.inclusion.sklearninclusion import using_sklearn

if using_sklearn:
    def test_ensemble():
        for f in ORBIT_LGT_SKATERS:
            err = hospital_mean_square_error_with_sporadic_fit(f=f, k=5, n=150, fit_frequency=100000, n_test=1)



if __name__=='__main__':
    test_ensemble()