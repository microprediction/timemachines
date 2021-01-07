from timemachines.skaters.pmd import pmd_auto
from timemachines.evaluation import quick_brown_fox, lazy_dog


def test_pmd_auto_univariate():
    err = quick_brown_fox(f=pmd_auto, n=500)


def test_pmd_auto_exogenous():
    err = lazy_dog(f=pmd_auto,n=500)