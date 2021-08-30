import random
from timemachines.skaters.smdk.smdkinclusion import using_simdkalman
from timemachines.skating import prior

if using_simdkalman:
    from timemachines.skaters.smdk.allsmdkskaters import SMDK_SKATERS
    from timemachines.skatertools.data.real import hospital

    def test_random_smdk_skaters():
        for _ in range(4):
            k = random.choice([1,3,5,11])
            n = 100
            f = random.choice(SMDK_SKATERS)
            y = hospital(n=n)
            e = [-1]*90+[100]+[-1]*90
            x, x_std = prior(f=f, y=y, k=k, e=e )


if __name__ == '__main__':
    import time
    st = time.time()
    test_random_smdk_skaters()
    print(time.time()-st)
