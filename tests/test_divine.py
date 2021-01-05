import numpy as np
from timemachines.thirdparty.divine import divine
from timemachines.conventions import rmse1


def test_divine():
    print(rmse1(f=divine))
