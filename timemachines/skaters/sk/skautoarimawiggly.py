from timemachines.skaters.sk.skinclusion import using_sktime

if using_sktime:
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
    from timemachines.skaters.sk.skautoarima import sk_autoarima
    from timemachines.skatertools.smoothing.wiggling import wiggler
    from timemachines.skatertools.combining.combiningforecasts import combine_using_mean,\
        combine_using_median, combine_using_huber

    def sk_autoarima_wiggly_factory(y: Y_TYPE, s: dict, k: int,
                                    a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None, m=5, d=0.1, combiner=None):
        return wiggler(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=sk_autoarima, m=m, d=d, combiner=combiner)


    def sk_autoarima_wiggly_mean_d05_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                        e: E_TYPE = None):
        return sk_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.05, combiner=combine_using_mean)


    def sk_autoarima_wiggly_median_d05_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                          e: E_TYPE = None):
        return sk_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.05, combiner=combine_using_median)


    def sk_autoarima_wiggly_huber_d05_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                         e: E_TYPE = None):
        return sk_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.05, combiner=combine_using_huber)


    def sk_autoarima_wiggly_mean_d05_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                         e: E_TYPE = None):
        return sk_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.05, combiner=combine_using_mean)


    def sk_autoarima_wiggly_median_d05_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                         e: E_TYPE = None):
        return sk_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.05, combiner=combine_using_median)


    def sk_autoarima_wiggly_huber_d05_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                         e: E_TYPE = None):
        return sk_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.05, combiner=combine_using_huber)

    SK_AUTOARIMA_WIGGLY_SKATERS = [sk_autoarima_wiggly_mean_d05_m3,
                                   sk_autoarima_wiggly_median_d05_m3,
                                   sk_autoarima_wiggly_huber_d05_m3,
                                   sk_autoarima_wiggly_mean_d05_m5,
                                   sk_autoarima_wiggly_median_d05_m5,
                                   sk_autoarima_wiggly_huber_d05_m5] # make more if you need them

else:
    SK_AUTOARIMA_WIGGLY_SKATERS = []
