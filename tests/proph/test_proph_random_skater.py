from timemachines.skatertools.data.real import hospital
import random
from timemachines.skaters.proph.prophiskaterfactory import using_prophet
from timemachines.skating import prior

if using_prophet:
    from timemachines.skaters.proph.allprophetskaters import PROPHET_SKATERS

    def test_random_prophet_skater(show=False):
        k = 3
        n = 100
        f = random.choice(PROPHET_SKATERS)
        print(f.__name__)
        y = hospital(n=n)
        e = [-1]*90+[100]+[-1]*90
        x, x_std = prior(f=f, y=y, k=k, e=e )


if __name__ == '__main__':
    import time
    st = time.time()
    test_random_prophet_skater()
    print(time.time()-st)
