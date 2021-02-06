from timemachines.skaters.simple.basic import BASIC_SKATERS
from timemachines.skaters.simple.linear import LINEAR_SKATERS
from timemachines.skaters.divine.divineskaters import DIVINE_SKATERS
from timemachines.skaters.proph.prophskaters import PROPHET_SKATERS
from timemachines.skaters.dlm.alldlmskaters import DLM_SKATERS


SKATERS = BASIC_SKATERS + LINEAR_SKATERS + DIVINE_SKATERS + PROPHET_SKATERS
SKATERS = BASIC_SKATERS + LINEAR_SKATERS

def skater_from_name(name):
    valid = [f for f in SKATERS if f.__name__==name ]
    return valid[0] if len(valid)==1 else None