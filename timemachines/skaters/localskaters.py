

# Local skaters are so named to differentiate them from those that require
# a connection (e.g. to look up Elo ratings and stack on the fly)

# Skaters have no hyper-parameters to set.
# Most of these are univariate and ignore y[1:] and a[:]
# We'll create some that use y[1:] more effectively as we go.

# If you add, please update skatertools.utilities.locations

# Home grown
from timemachines.skaters.simple.movingaverage import EMA_SKATERS
from timemachines.skaters.simple.linear import LINEAR_SKATERS
from timemachines.skaters.simple.thinking import THINKING_SKATERS
from timemachines.skaters.simple.hypocraticensemble import HYPOCRATIC_ENSEMBLE_SKATERS
# Using 3rd party
from timemachines.skaters.divine.divineskaters import DIVINE_SKATERS
from timemachines.skaters.proph.allprophetskaters import PROPHET_SKATERS
from timemachines.skaters.dlm.alldlmskaters import DLM_SKATERS
from timemachines.skaters.pmd.allpmdskaters import PMD_SKATERS
from timemachines.skaters.tsa.alltsaskaters import TSA_SKATERS
from timemachines.skaters.nproph.allnprophetskaters import NPROPHET_SKATERS
from timemachines.skaters.orbt.allorbitskaters import ORBIT_SKATERS
from timemachines.skaters.bats.allbatsskaters import BATS_SKATERS
from timemachines.skaters.rvr.allriverskaters import RIVER_SKATERS
from timemachines.skaters.sk.allskskaters import SK_SKATERS
from timemachines.skaters.gk.allgreykiteskaters import GREYKITE_SKATERS
from timemachines.skaters.smdk.allsmdkskaters import SMDK_SKATERS
from timemachines.skaters.drts.alldartsskaters import DARTS_SKATERS
from timemachines.skaters.mrln.allmerlionskaters import MERLION_SKATERS

LOCAL_SKATERS = EMA_SKATERS + PROPHET_SKATERS + DIVINE_SKATERS + DLM_SKATERS + \
                THINKING_SKATERS + PMD_SKATERS + TSA_SKATERS + NPROPHET_SKATERS + \
                HYPOCRATIC_ENSEMBLE_SKATERS + ORBIT_SKATERS + BATS_SKATERS \
                + RIVER_SKATERS + SK_SKATERS + GREYKITE_SKATERS + SMDK_SKATERS \
                + DARTS_SKATERS + MERLION_SKATERS


LEFT_OUT_FOR_NOW = LINEAR_SKATERS # + ...

# Skaters designed for online use
FAST_LOCAL_SKATERS = EMA_SKATERS + THINKING_SKATERS + HYPOCRATIC_ENSEMBLE_SKATERS + RIVER_SKATERS + SMDK_SKATERS



def local_skater_from_name(name):
    valid = [f for f in LOCAL_SKATERS if f.__name__ == name]
    return valid[0] if len(valid)==1 else None




