from timemachines.skaters.simple.movingaverage import EMA_SKATERS, EMA_R1_SKATERS
from timemachines.skaters.simple.linear import LINEAR_SKATERS
from timemachines.skaters.divine.divineskaters import DIVINE_SKATERS
from timemachines.skaters.proph.allprophetskaters import PROPHET_SKATERS, PROPHET_R2_SKATERS
from timemachines.skaters.dlm.alldlmskaters import DLM_SKATERS
from timemachines.skaters.simple.thinking import THINKING_SKATERS
from timemachines.skaters.simple.hypocraticensemble import HYPOCRATIC_ENSEMBLE_SKATERS
from timemachines.skaters.pmd.allpmdskaters import PMD_SKATERS
from timemachines.skaters.tsa.alltsaskaters import TSA_SKATERS
from timemachines.skaters.nproph.allnprophetskaters import NPROPHET_SKATERS
from timemachines.skaters.uberorbit.allorbitskaters import ORBIT_SKATERS
from timemachines.skaters.bats.allbatsskaters import BATS_SKATERS
from timemachines.skaters.rvr.allriverskaters import RIVER_SKATERS
from timemachines.skaters.sk.allskskaters import SK_SKATERS
from timemachines.skaters.linkedingreykite.alllinkedingreykiteskaters import LINKEDINGREYKITE_SKATERS
from timemachines.skaters.smdk.allsmdkskaters import SMDK_SKATERS

# Local skaters don't access the Elo ratings

# Listing of skaters with no hyper-parameters
LOCAL_SKATERS = EMA_SKATERS + PROPHET_SKATERS + DIVINE_SKATERS + DLM_SKATERS + \
                THINKING_SKATERS + PMD_SKATERS + TSA_SKATERS + NPROPHET_SKATERS + \
                HYPOCRATIC_ENSEMBLE_SKATERS + ORBIT_SKATERS + BATS_SKATERS \
                + RIVER_SKATERS + SK_SKATERS + LINKEDINGREYKITE_SKATERS + SMDK_SKATERS


LEFT_OUT_FOR_NOW = LINEAR_SKATERS # + ...

# Skaters designed for online use
FAST_SKATERS = EMA_SKATERS + THINKING_SKATERS + HYPOCRATIC_ENSEMBLE_SKATERS + RIVER_SKATERS + SMDK_SKATERS


# Some skaters with hyper-parameters for tuning
SKATERS_R3 = []
SKATERS_R2 = PROPHET_R2_SKATERS
SKATERS_R1 = EMA_R1_SKATERS


def local_skater_from_name(name):
    valid = [f for f in LOCAL_SKATERS if f.__name__ == name]
    return valid[0] if len(valid)==1 else None




