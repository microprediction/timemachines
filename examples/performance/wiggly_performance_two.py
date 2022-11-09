from timemachines.skatertools.comparison.performancerating import performance_rating


if __name__=='__main__':
    from timemachines.skaters.sk.sfautoarimawiggly import sf_autoarima_wiggly_d050_m1, \
        sf_autoarima_wiggly_d050_m2, sf_autoarima_wiggly_d010_m2
    from timemachines.skaters.sk.sfautoarima import sf_autoarima
    skaters = [ sf_autoarima,
                sf_autoarima_wiggly_d050_m1,
                sf_autoarima_wiggly_d050_m2,
                sf_autoarima_wiggly_d010_m2 ]
    skater_names = [f.__name__ for f in skaters ]
    performance_rating(skater_names=skater_names, update_frequency=1)