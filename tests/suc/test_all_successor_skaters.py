from timemachines.skaters.suc.successorinclusion import using_successor
from timemachines.skating import prior

if using_successor:
    from successor.skaters.allsuccessorskaters import SUCCESSOR_SKATERS
    from timemachines.skatertools.data.real import hospital

    def test_all_successor_skaters():
            for k in [3,1,11]:
                n = 100
                for f in SUCCESSOR_SKATERS:
                    y = hospital(n=n)
                    e = [-1]*90+[100]+[-1]*90
                    x, x_std = prior(f=f, y=y, k=k, e=e )


if __name__ == '__main__':
    import time
    st = time.time()
    test_all_successor_skaters()
    print(time.time()-st)
