from timemachines.skatertools.data.real import hospital
import random
from timemachines.skaters.drts.dartsinclusion import using_darts
from timemachines.skating import prior

if using_darts:
    from timemachines.skaters.drts.alldartsskaters import DARTS_SKATERS

    def test_random_skater(show=False):
        k = 3
        n = 100
        f = random.choice(DARTS_SKATERS)
        y = hospital(n=n)
        neg_y = [-yt for yt in y]
        e = [-1]*90+[100]+[-1]*90
        x, x_std = prior(f=f, y=neg_y, k=k, e=e )


if __name__ == '__main__':
    assert using_darts, 'pip install darts'
    import time
    for _ in range(100):
        st = time.time()
        test_random_skater()
        print(time.time()-st)
