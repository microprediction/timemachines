from timemachines.skatertools.data.real import hospital
import random
from timemachines.skaters.kts.ktsinclusion import using_kats
from timemachines.skating import prior

if using_kats:
    from timemachines.skaters.kts.allkatsskaters import KATS_SKATERS

    def test_random_skater(show=False):
        k = 3
        n = 100
        f = random.choice(KATS_SKATERS)
        print( f.__name__ )
        y = hospital(n=n)
        neg_y = [-yt for yt in y]
        e = [-1]*90+[100]+[-1]*90
        x, x_std = prior(f=f, y=neg_y, k=k, e=e )


if __name__ == '__main__':
    assert using_kats, 'pip install kats'
    import time
    for _ in range(100):
        st = time.time()
        test_random_skater()
        print(time.time()-st)
