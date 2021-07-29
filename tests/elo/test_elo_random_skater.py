import random
from timemachines.skaters.elo.alleloskaters import ELO_SKATERS
from timemachines.skating import prior
from timemachines.skatertools.data.real import hospital
from timemachines.skatertools.utilities.internet import connected_to_internet

if connected_to_internet():

    def test_random_elo_skater():
        k = random.choice([1,3,5,11])
        n = 100
        f = random.choice(ELO_SKATERS)
        y = hospital(n=n)
        e = [-1]*90+[100]+[-1]*90
        x, x_std = prior(f=f, y=y, k=k, e=e )


if __name__ == '__main__':
    import time
    st = time.time()
    test_random_elo_skater()
    print(time.time()-st)
