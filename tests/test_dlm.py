from timemachines.skaters.dlm import dlm_seasonal
from timemachines.evaluation import quick_brown_fox, lazy_dog


def test_dlm_seasonal_univariate():
    err = quick_brown_fox(f=dlm_seasonal, n=150, n_burn=44)


def test_dlm_seasonal_exogenous():
    err = lazy_dog(f=dlm_seasonal, n=150, n_burn=44)