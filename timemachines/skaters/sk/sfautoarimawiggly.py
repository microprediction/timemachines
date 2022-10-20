from timemachines.skaters.sk.sfinclusion import using_statsforecast

if using_statsforecast:
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
    from timemachines.skaters.sk.sfautoarima import sf_autoarima
    from timemachines.skatertools.smoothing.wiggling import wiggler


    def sf_autoarima_wiggly_factory(y: Y_TYPE, s: dict, k: int,
                                    a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None, m=5, d=0.1):
        return wiggler(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=sf_autoarima, m=m, d=d)


    def sf_autoarima_wiggly_d001_m1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=1, d=0.01)


    def sf_autoarima_wiggly_d001_m2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=2, d=0.01)


    def sf_autoarima_wiggly_d001_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.01)


    def sf_autoarima_wiggly_d001_m4(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=4, d=0.01)


    def sf_autoarima_wiggly_d001_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.01)


    def sf_autoarima_wiggly_d010_m1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=1, d=0.1)

    def sf_autoarima_wiggly_d010_m2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=2, d=0.1)

    def sf_autoarima_wiggly_d010_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.1)

    def sf_autoarima_wiggly_d010_m4(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=4, d=0.1)

    def sf_autoarima_wiggly_d010_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.1)


    def sf_autoarima_wiggly_d050_m1(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=1, d=0.5)


    def sf_autoarima_wiggly_d050_m2(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=2, d=0.5)


    def sf_autoarima_wiggly_d050_m3(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=3, d=0.5)


    def sf_autoarima_wiggly_d050_m4(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=4, d=0.5)


    def sf_autoarima_wiggly_d050_m5(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None,
                                    e: E_TYPE = None):
        return sf_autoarima_wiggly_factory(y=y, s=s, k=k, a=a, t=t, e=e, m=5, d=0.5)

    SF_AUTOARIMA_WIGGLY_SKATERS = [ sf_autoarima_wiggly_d001_m1, sf_autoarima_wiggly_d001_m2,
                                sf_autoarima_wiggly_d001_m3, sf_autoarima_wiggly_d001_m4,
                                sf_autoarima_wiggly_d001_m5,
                                sf_autoarima_wiggly_d010_m1, sf_autoarima_wiggly_d010_m2,
                                sf_autoarima_wiggly_d010_m3, sf_autoarima_wiggly_d010_m4,
                                sf_autoarima_wiggly_d010_m5,
                                sf_autoarima_wiggly_d050_m1, sf_autoarima_wiggly_d050_m2,
                                sf_autoarima_wiggly_d050_m3, sf_autoarima_wiggly_d050_m4,
                                sf_autoarima_wiggly_d050_m5 ]

else:
    SF_AUTOARIMA_WIGGLY_SKATERS = []
