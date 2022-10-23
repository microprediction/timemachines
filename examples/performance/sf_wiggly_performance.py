from timemachines.skatertools.comparison.performancerating import performance_rating


if __name__=='__main__':
    from timemachines.skaters.sk.sfautoarimawiggly import SF_AUTOARIMA_WIGGLY_SKATERS
    from timemachines.skaters.sk.sfautoarima import sf_autoarima
    skater_names = [f.__name__ for f in SF_AUTOARIMA_WIGGLY_SKATERS + [sf_autoarima] ]
    performance_rating(skater_names=skater_names, update_frequency=1)