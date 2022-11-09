from timemachines.skaters.sk.sfinclusion import using_statsforecast

if using_statsforecast:
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
    from timemachines.skaters.sk.sfautoarima import sf_autoarima
    from timemachines.skatertools.smoothing.wiggling import wiggler
    from timemachines.skatertools.combining.combiningforecasts import combine_using_mean,\
        combine_using_median, combine_using_huber

    def sf_autoarima_wiggly_factory(y: Y_TYPE, s: dict, k: int,
                                    a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None, m=5, d=0.1, combiner=None):
        return wiggler(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=sf_autoarima, m=m, d=d, combiner=combiner)


    def sf_autoarima_wiggly_mean_d05_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                        e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.05, combiner=combine_using_mean)


    def sf_autoarima_wiggly_median_d05_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                          e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.05, combiner=combine_using_median)


    def sf_autoarima_wiggly_huber_d05_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                         e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.05, combiner=combine_using_huber)


    def sf_autoarima_wiggly_mean_d05_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                         e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.05, combiner=combine_using_mean)


    def sf_autoarima_wiggly_median_d05_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                         e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.05, combiner=combine_using_median)


    def sf_autoarima_wiggly_huber_d05_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                         e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.05, combiner=combine_using_huber)

    SF_AUTOARIMA_WIGGLY_SKATERS = [sf_autoarima_wiggly_mean_d05_m3,
                                   sf_autoarima_wiggly_median_d05_m3,
                                   sf_autoarima_wiggly_huber_d05_m3,
                                   sf_autoarima_wiggly_mean_d05_m5,
                                   sf_autoarima_wiggly_median_d05_m5,
                                   sf_autoarima_wiggly_huber_d05_m5] # make more if you need them

else:
    SF_AUTOARIMA_WIGGLY_SKATERS = []
