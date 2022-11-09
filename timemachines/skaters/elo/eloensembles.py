from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, T_TYPE, E_TYPE
from timemachines.skatertools.ensembling.ensemblefactory import R_BALANCED, R_AGGRESSIVE, R_PRECISION
from timemachines.skaters.elo.eloensemblefactory import elo_ensemble_factory
from timemachines.skatertools.utilities.internet import connected_to_internet

FASTEST = 1.0
FASTER = 10.0

if connected_to_internet():

    # Fastest

    def elo_fastest_mixed_precision_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_PRECISION, n=15, max_seconds=FASTEST,
                                    categories=['univariate','special','residual'])

    def elo_fastest_mixed_balanced_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_BALANCED, n=15, max_seconds=FASTEST,
                                    categories=['univariate','special','residual'])

    def elo_fastest_mixed_aggressive_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_AGGRESSIVE, n=15, max_seconds=FASTEST,
                                    categories=['univariate','special','residual'])

   # Faster

    def elo_faster_mixed_precision_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_PRECISION, n=15, max_seconds=FASTER,
                                    categories=['univariate','special','residual'])

    def elo_faster_mixed_balanced_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_BALANCED, n=15, max_seconds=FASTER,
                                    categories=['univariate','special','residual'])

    def elo_faster_mixed_aggressive_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_AGGRESSIVE, n=15, max_seconds=FASTER,
                                    categories=['univariate','special','residual'])


    # Univariate only


    def elo_fastest_univariate_precision_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_PRECISION, n=5, max_seconds=FASTEST, category='univariate')


    def elo_fastest_univariate_balanced_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_BALANCED, n=5, max_seconds=FASTEST, category='univariate')


    def elo_fastest_univariate_aggressive_ensemble(y: Y_TYPE, s: dict, k: int = 1, a:A_TYPE = None, t:T_TYPE = None, e:E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_AGGRESSIVE, n=5, max_seconds=FASTEST, category='univariate')

   # Faster

    def elo_faster_univariate_precision_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_PRECISION, n=5, max_seconds=FASTER, category='univariate')


    def elo_faster_univariate_balanced_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None,
                                                 e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_BALANCED, n=5, max_seconds=FASTER, category='univariate')


    def elo_faster_univariate_aggressive_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_AGGRESSIVE, n=5, max_seconds=FASTER, category='univariate')


    # Residual only

    def elo_fastest_residual_precision_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None,
                                                  e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_PRECISION, n=5, max_seconds=FASTEST, category='residual')


    def elo_fastest_residual_balanced_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None,
                                                 e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_BALANCED, n=5, max_seconds=FASTEST, category='residual')


    def elo_fastest_residual_aggressive_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None,
                                                   e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_AGGRESSIVE, n=5, max_seconds=FASTEST, category='residual')


    def elo_faster_residual_precision_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None,
                                                 e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_PRECISION, n=5, max_seconds=FASTER, category='residual')


    def elo_faster_residual_balanced_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None,
                                                e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_BALANCED, n=5, max_seconds=FASTER, category='residual')


    def elo_faster_residual_aggressive_ensemble(y: Y_TYPE, s: dict, k: int = 1, a: A_TYPE = None, t: T_TYPE = None,
                                                  e: E_TYPE = None):
        return elo_ensemble_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=R_AGGRESSIVE, n=5, max_seconds=FASTER, category='residual')


    ELO_MIXED_SKATERS = [elo_fastest_mixed_aggressive_ensemble, elo_fastest_mixed_balanced_ensemble,
                              elo_fastest_mixed_precision_ensemble,
                              elo_faster_mixed_aggressive_ensemble, elo_faster_mixed_precision_ensemble,
                              elo_faster_mixed_balanced_ensemble]

    ELO_UNIVARIATE_SKATERS = [ elo_fastest_univariate_aggressive_ensemble, elo_fastest_univariate_balanced_ensemble,
                             elo_fastest_univariate_precision_ensemble,
                             elo_faster_univariate_aggressive_ensemble, elo_faster_univariate_precision_ensemble,
                             elo_faster_univariate_balanced_ensemble]

    ELO_RESIDUAL_SKATERS = [elo_fastest_residual_aggressive_ensemble, elo_fastest_residual_balanced_ensemble,
                              elo_fastest_residual_precision_ensemble,
                              elo_faster_residual_aggressive_ensemble, elo_faster_residual_precision_ensemble,
                              elo_faster_residual_balanced_ensemble]

    ELO_ENSEMBLE_SKATERS = ELO_UNIVARIATE_SKATERS + ELO_RESIDUAL_SKATERS + ELO_MIXED_SKATERS

else:
    ELO_ENSEMBLE_SKATERS = []

