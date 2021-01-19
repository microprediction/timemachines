from timemachines.skaters.dlmseasonal import dlm_seasonal
from timemachines.evaluation import quick_brown_fox, lazy_dog
import numpy as np


def test_dlm_seasonal_univariate():
    print(quick_brown_fox(f=dlm_seasonal, n=150, n_burn=44, r=np.random.rand()))


def test_dlm_seasonal_exogenous():
    print(lazy_dog(f=dlm_seasonal, n=150, n_burn=44, r=np.random.rand()))


if __name__=='__main__':
    test_dlm_seasonal_exogenous()