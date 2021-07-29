from timemachines.skaters.elo.alleloskaters import ELO_SKATERS

NETWORKED_SKATERS = ELO_SKATERS

# Networked skaters access Elo ratings


def networked_skater_from_name(name):
    valid = [f for f in NETWORKED_SKATERS if f.__name__ == name]
    return valid[0] if len(valid)==1 else None
