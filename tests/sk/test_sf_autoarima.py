from timemachines.skatertools.data.real import hospital
import random
from timemachines.skaters.sk.skinclusion import using_sktime
from timemachines.skaters.sk.sfinclusion import using_statsforecast
from timemachines.skating import prior

if using_sktime and using_statsforecast:
    from timemachines.skaters.sk.sfautoarima import sf_autoarima

    def test_sf_autoarima(show=False):
        k = 3
        n = 100
        f = sf_autoarima
        y = hospital(n=n)
        neg_y = [-yt for yt in y[:20]]
        e = [-1]*15+[100]+[-1]*90
        x, x_std = prior(f=f, y=neg_y, k=k, e=e )


if __name__ == '__main__':
    import time
    for _ in range(2):
        st = time.time()
        test_sf_autoarima(show=True)
        print(time.time()-st)
