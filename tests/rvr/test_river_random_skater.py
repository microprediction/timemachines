import random
from timemachines.skaters.rvr.rvrinclusion import using_river
from timemachines.skating import prior

if using_river:
    from timemachines.skaters.rvr.allriverskaters import RIVER_SKATERS
    from timemachines.skatertools.data.real import hospital

    def test_random_river_skaters():
        # River is fast :)
        for _ in range(20):
            k = random.choice([1,3,5,11])
            n = 100
            f = random.choice(RIVER_SKATERS)
            y = hospital(n=n)
            e = [-1]*90+[100]+[-1]*90
            x, x_std = prior(f=f, y=y, k=k, e=e )


if __name__ == '__main__':
    import time
    st = time.time()
    test_random_river_skaters()
    print(time.time()-st)
