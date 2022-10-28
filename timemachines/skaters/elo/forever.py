from timemachines.skatertools.utilities.internet import connected_to_internet
from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, T_TYPE, E_TYPE

FASTEST = 1.0
FASTER = 10.0

if connected_to_internet():

    from timemachines.skaters.elo.eloensembles import elo_fastest_mixed_aggressive_ensemble
    forever = elo_fastest_mixed_aggressive_ensemble

    FOREVER_SKATERS = [ forever ]

else:

    FOREVER_SKATERS = []

