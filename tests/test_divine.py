import numpy as np
from timemachines.thirdparty.divine import divinity_f1
from timemachines.conventions import rmse1


def test_divine():
    print( rmse1(f=divinity_f1) )
