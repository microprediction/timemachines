from timemachines.machines.pmd import pmd_auto
from timemachines.metrics import brownian_rmse, exogenous_rmse


def test_pmd_auto_univariate():
    err = brownian_rmse(f=pmd_auto,n=500)


def test_pmd_auto_exogenous():
    err = exogenous_rmse(f=pmd_auto,n=500)