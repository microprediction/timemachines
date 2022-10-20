from timemachines.skatertools.data.ornstein import simulate_arima_like_path
from timemachines.skatertools.sensitivity.skatersensitivity import skater_bump


def skater_bump_plot(f, g, ndx, k):
    """ Plot sensitivity to k'th to last observation,
        and compare to an alternative g that might be smoother
    """
    for _ in range(200):
        import numpy as np
        ys = simulate_arima_like_path(seq_len=50)
        y_final, x_final = skater_bump(ys=ys, f=f, ndx=ndx, k=k)
        discont_max = np.max(np.diff(np.array(x_final)))
        discont_median = np.median(np.abs(np.diff(np.array(x_final))))
        if discont_max>5*discont_median:
            print('Comparing ...')
            y_alt, x_alt = skater_bump(ys=ys, f=g, ndx=ndx, k=k)
            import matplotlib.pyplot as plt
            plt.plot(y_final,x_final, 'rx')
            plt.plot(y_alt, x_alt, 'go')
            plt.ylabel('Prediction '+str(k)+' steps ahead')
            kstub = g.__name__.split('_')[-1]
            plt.xlabel('Value taken by y['+str(ndx)+'] w/ wiggle '+kstub)
            plt.grid()
            plt.title(f.__name__)
            plt.legend(['original','wiggled'])
            plt.show()
            import time
            time.sleep(2)
            plt.close()


if __name__=='__main__':
    from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    from timemachines.skaters.sk.sfautoarimawiggly import sf_autoarima_wiggly_d010_m2 as g
    from timemachines.skaters.proph.prophskaterssingular import fbprophet_univariate as f
    #from timemachines.skaters.simple.thinking import thinking_fast_and_slow as f
    #from timemachines.skaters.simple.movingaverage import precision_ema_ensemble as f
    from timemachines.skaters.tsa.tsatheta import tsa_theta_additive as f
    from timemachines.skaters.sk.sfautoarima import sf_autoarima as f

    skater_bump_plot(f=f, g=g, ndx=-5, k=1)

