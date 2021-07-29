from timemachines.skaters.localskaters import LOCAL_SKATERS, local_skater_from_name
from timemachines.skaters.networkedskaters import NETWORKED_SKATERS

SKATERS = NETWORKED_SKATERS + LOCAL_SKATERS


def skater_from_name(name):
    valid = [f for f in SKATERS if f.__name__ == name]
    return valid[0] if len(valid)==1 else None

