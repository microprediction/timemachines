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

SKATERS_R3 = []
SKATERS_R2 = PROPHET_R2_SKATERS
SKATERS_R1 = EMA_R1_SKATERS

# Leaving out LINEAR_SKATERS

# And with no hyper-params...
SKATERS = EMA_SKATERS + PROPHET_SKATERS + DIVINE_SKATERS + DLM_SKATERS + \
          THINKING_SKATERS + PMD_SKATERS + TSA_SKATERS + NPROPHET_SKATERS + HYPOCRATIC_ENSEMBLE_SKATERS


def skater_from_name(name):
    valid = [f for f in SKATERS if f.__name__==name ]
    return valid[0] if len(valid)==1 else None


if __name__=='__main__':
    from pprint import pprint
    pprint([sk.__name__ for sk in SKATERS])
    print(len(SKATERS))