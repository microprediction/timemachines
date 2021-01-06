from timemachines.machines.divine import divine
from timemachines.metrics import brownian_rmse


def test_divine():
    err = brownian_rmse(f=divine)

