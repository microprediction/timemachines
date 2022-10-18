from timemachines.skatertools.data.real import hospital
import random
from timemachines.skaters.sk.skinclusion import using_sktime
from timemachines.skaters.sk.sfinclusion import using_statsforecast
from timemachines.skating import prior

if using_sktime and using_statsforecast:
    from timemachines.skaters.sk.sfautoarima import SF_AA_SKATERS
    from timemachines.skatertools.smoothing.wiggling import wiggler

    def test_random_skater_smooth(show=False):
        k = 3
        n = 15
        f = random.choice(SF_AA_SKATERS)

        def g(y,s,k):
            return wiggler(f=f, y=y, s=s, k=k, m=2)

        y = hospital(n=n)

        s = {}
        for yi in y:
            x, x_std, s = g(y=yi, s=s, k=k)

if __name__ == '__main__':
    import time
    for _ in range(2):
        st = time.time()
        test_random_skater_smooth()
        print(time.time()-st)
