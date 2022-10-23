from timemachines.skatertools.comparison.performancerating import performance_rating


if __name__=='__main__':
    from timemachines.skaters.sk.skautoarimawiggly import sk_autoarima_wiggly_mean_d05_m3, \
        sk_autoarima_wiggly_huber_d05_m3
    from timemachines.skaters.sk.sfautoarimawiggly import sf_autoarima_wiggly_mean_d05_m3, \
        sf_autoarima_wiggly_huber_d05_m3
    from timemachines.skaters.sk.skautoarima import sk_autoarima
    from timemachines.skaters.sk.sfautoarima import sf_autoarima

    skater_names = [f.__name__ for f in  [sk_autoarima,
                                          sf_autoarima,
                                          sk_autoarima_wiggly_huber_d05_m3,
                                          sk_autoarima_wiggly_mean_d05_m3,
                                          sf_autoarima_wiggly_mean_d05_m3,
                                          sf_autoarima_wiggly_huber_d05_m3] ]
    performance_rating(skater_names=skater_names, update_frequency=1)