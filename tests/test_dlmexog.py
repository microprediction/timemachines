from timemachines.skaters.dlmexog import dlm_exog
from timemachines.evaluation import quick_brown_fox, lazy_dog
import numpy as np


def test_dlm_exog_univariate():
    print(quick_brown_fox(f=dlm_exog, n=150, n_burn=44))


def test_dlm_exog_exogenous():
    print(lazy_dog(f=dlm_exog, n=150, n_burn=44, r=np.random.rand()))


if __name__=='__main__':
    test_dlm_exog_univariate()