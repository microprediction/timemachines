from timemachines.skaters.dlmauto import dlm_auto
from timemachines.evaluation import quick_brown_fox, lazy_dog


def test_dlm_auto_univariate():
    print(quick_brown_fox(f=dlm_auto, n=150, n_burn=44))


def test_dlm_auto_exogenous():
    err = lazy_dog(f=dlm_auto, n=150, n_burn=44)