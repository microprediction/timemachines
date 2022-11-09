from timemachines.skatertools.visualization.skaterbumpplot import skater_bump_plot


if __name__=='__main__':
    #from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    #from timemachines.skaters.simple.thinking import thinking_fast_and_slow as f
    #from timemachines.skaters.simple.movingaverage import precision_ema_ensemble as f
    #from timemachines.skaters.tsa.tsatheta import tsa_theta_additive as f
    from timemachines.skaters.sk.sfautoarima import sf_autoarima as f
    #from timemachines.skaters.proph.prophskaterssingular import fbprophet_univariate as f

    skater_bump_plot(f=f, ndx=-3, k=1)

