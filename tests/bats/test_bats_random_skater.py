from timemachines.skatertools.data.real import hospital
import random
from timemachines.skaters.bats.batsinclusion import using_bats
from timemachines.skating import prior

if using_bats:
    from timemachines.skaters.bats.allbatsskaters import BATS_SKATERS

    def test_random_skater(show=False):
        k = 3
        n = 100
        f = random.choice(BATS_SKATERS)
        y = hospital(n=n)
        e = [-1]*90+[100]+[-1]*90
        x, x_std = prior(f=f, y=y, k=k, e=e )


if __name__ == '__main__':
    import time
    st = time.time()
    test_random_skater()
    print(time.time()-st)
