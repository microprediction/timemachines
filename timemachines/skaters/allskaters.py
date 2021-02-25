from timemachines.skaters.simple.empirical import BASIC_SKATERS, BASIC_R1_SKATERS
from timemachines.skaters.simple.linear import LINEAR_SKATERS
from timemachines.skaters.divine.divineskaters import DIVINE_SKATERS
from timemachines.skaters.proph.allprophetskaters import PROPHET_SKATERS, PROPHET_R2_SKATERS
from timemachines.skaters.dlm.alldlmskaters import DLM_SKATERS
from timemachines.skaters.simple.thinking import THINKING_SKATERS


SKATERS_R3 = []
SKATERS_R2 = PROPHET_R2_SKATERS
SKATERS_R1 = BASIC_R1_SKATERS

# And with no hyper-params...
SKATERS = BASIC_SKATERS + LINEAR_SKATERS + PROPHET_SKATERS + DIVINE_SKATERS + DLM_SKATERS + THINKING_SKATERS


def skater_from_name(name):
    valid = [f for f in SKATERS if f.__name__==name ]
    return valid[0] if len(valid)==1 else None