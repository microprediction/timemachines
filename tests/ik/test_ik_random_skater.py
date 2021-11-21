from timemachines.skatertools.data.real import hospital
import random
from timemachines.skaters.ik.ikinclusion import using_ik
from timemachines.skating import prior

if using_ik:
    from timemachines.skaters.ik.allikskaters import IK_SKATERS

    def test_random_skater(show=False):
        k = 3
        n = 100
        f = random.choice(IK_SKATERS)
        print( f.__name__ )
        y = hospital(n=n)
        neg_y = [-yt for yt in y]
        e = [-1]*90+[100]+[-1]*90
        t = 10
        r = 3
        x, x_std = prior(f=f, y=neg_y, k=k, t=t, e=e, r=r)


if __name__ == '__main__':
    assert using_ik, 'pip install -U scikit-learn'
    import time
    for _ in range(100):
        st = time.time()
        test_random_skater()
        print(time.time()-st)
