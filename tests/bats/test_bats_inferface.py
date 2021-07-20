from timemachines.skatertools.data.real import hospital_with_exog, hospital
import random
from timemachines.skaters.bats.batsinclusion import using_bats

if using_bats:
    from timemachines.skaters.bats.batsifactory import bats_iskater_factory
    from timemachines.skaters.bats.allbatsskaters import BATS_SKATERS

    def test_univariate_without_time(show=False):
        k = 3
        n = 100
        f = random.choice(BATS_SKATERS)
        y = hospital(n=n)
        x, x_std = bats_iskater_factory(y=y, k=k)


if __name__ == '__main__':
    import time
    st = time.time()
    test_univariate_without_time()
    print(time.time()-st)
